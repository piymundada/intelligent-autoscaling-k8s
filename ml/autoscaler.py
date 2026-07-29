"""
JVM-Aware Predictive Autoscaler (thesis §3.7.2)

Control loop:
  1. Every CONTROL_INTERVAL seconds, fetch the latest metric window from Prometheus
  2. Feed the window into the trained LSTM to predict workload PRED_HORIZON steps ahead
  3. Convert the predicted RPS to a replica recommendation
  4. Apply safeguards (max step change, stabilisation window)
  5. Scale the Kubernetes Deployment if the recommendation differs from current replicas
  6. Log all decisions to a CSV for later comparison with the HPA baseline

Usage:
    python3.11 autoscaler.py \
        --prometheus-url http://localhost:9090 \
        --model models/lstm_forecaster.pt \
        --scaler data/processed/scaler.json \
        --namespace thesis \
        --deployment spring-boot-app \
        --rps-per-replica 15 \
        --min-replicas 1 \
        --max-replicas 4 \
        --control-interval 30 \
        --log-file logs/predictive_autoscaler.csv
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import torch

from config import (
    AUTOSCALER_FEATURE_QUERIES,
    JVM_FLOOR_CPU_HIGH_PCT,
    JVM_FLOOR_GC_HIGH_MS,
    JVM_FLOOR_GC_SEVERE_MS,
    JVM_FLOOR_MAX_BUMP,
    RPS_PER_REPLICA,
    SEQ_LEN,
    WARMUP_GRACE_CYCLES,
    jvm_floor_bump,
)
from lstm_model import LSTMForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("predictive-autoscaler")


def query_instant(prometheus_url: str, query: str) -> float:
    """Query Prometheus for a single scalar value at the current moment."""
    resp = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={"query": query},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    result = data["data"]["result"]
    if not result:
        return float("nan")
    return float(result[0]["value"][1])


def get_current_replicas(namespace: str, deployment: str) -> int:
    out = subprocess.check_output(
        ["kubectl", "get", "deployment", deployment,
         "-n", namespace, "-o", "jsonpath={.spec.replicas}"],
        text=True,
    )
    return int(out.strip() or "1")


def set_replicas(namespace: str, deployment: str, replicas: int) -> None:
    subprocess.check_call(
        ["kubectl", "scale", "deployment", deployment,
         "-n", namespace, f"--replicas={replicas}"],
    )


def rps_to_replicas(
    predicted_rps: float,
    rps_per_replica: float,
    min_r: int,
    max_r: int,
) -> int:
    """Convert predicted RPS to a replica count with 50% headroom buffer.
    The headroom provisions ahead of the gradual ramp so capacity is ready before
    the peak (and before a single pod is overwhelmed enough to stop reporting)."""
    raw = predicted_rps / rps_per_replica
    with_headroom = raw * 1.50
    return max(min_r, min(max_r, int(np.ceil(with_headroom))))


class PredictiveAutoscaler:
    def __init__(self, args: argparse.Namespace):
        self.prometheus_url = args.prometheus_url
        self.namespace = args.namespace
        self.deployment = args.deployment
        self.rps_per_replica = args.rps_per_replica
        self.min_replicas = args.min_replicas
        self.max_replicas = args.max_replicas
        self.control_interval = args.control_interval
        self.max_step = args.max_step_change
        self.scale_down_delay = args.scale_down_delay_cycles

        # JVM-pressure floor (the JVM-aware contribution). Conservative + tunable;
        # tune thresholds with ml/backtest_jvm_floor.py before live runs. Disable
        # with --no-jvm-floor to run the RPS-only ablation arm.
        self.jvm_floor_enabled = args.jvm_floor
        self.gc_high = args.gc_high
        self.gc_severe = args.gc_severe
        self.cpu_high = args.cpu_high
        self.jvm_floor_max_bump = args.jvm_floor_max_bump
        # Warm-up grace: suppress the CPU/GC floor for the first N control cycles
        # while the JVM JIT-compiles (cold-start CPU is a fraction of few container
        # cores -> reads 60-95% and falsely trips the floor). Hardware/env-specific:
        # 0 locally (4-core Minikube warmup stays <60%), ~3 on EKS (1 container core).
        # Separates transient warmup from sustained load by TIME, keeping the floor
        # fully sensitive under real load. See k8s/eks/autoscaler.yaml.
        self.warmup_grace = args.warmup_grace_cycles
        self._step_count = 0

        # Load scaler parameters
        with open(args.scaler) as f:
            self.scaler_params = json.load(f)
        self.feature_names = self.scaler_params["feature_names"]
        self.data_min = np.array(self.scaler_params["data_min_"])
        self.data_max = np.array(self.scaler_params["data_max_"])
        self.seq_len = self.scaler_params["seq_len"]
        self.target_idx = self.scaler_params["target_idx"]

        # Load LSTM model
        device = torch.device("cpu")
        cfg = self.scaler_params.get("model_config", {})
        self.model = LSTMForecaster(
            input_size=len(self.feature_names),
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        ).to(device)
        self.model.load_state_dict(torch.load(args.model, map_location=device))
        self.model.eval()
        self.device = device

        # Rolling window of normalised metric vectors
        self.window: deque = deque(maxlen=self.seq_len)

        # Stabilisation counter: suppress scale-down until enough idle cycles
        self.cycles_since_scaleup: int = 0

        # CSV decision log
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        self._log_file = open(args.log_file, "w", newline="")
        self._csv = csv.DictWriter(self._log_file, fieldnames=[
            "timestamp", "current_replicas", "predicted_rps",
            "rps_replicas", "jvm_floor_bump",
            "recommended_replicas", "applied_replicas", "action",
        ] + self.feature_names)
        self._csv.writeheader()

    def _fetch_metrics(self) -> dict:
        """Fetch all feature metrics from Prometheus (instant query)."""
        row = {}
        for name in self.feature_names:
            query = AUTOSCALER_FEATURE_QUERIES.get(name)
            if query:
                row[name] = query_instant(self.prometheus_url, query)
            else:
                row[name] = float("nan")
        return row

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        denom = self.data_max - self.data_min
        denom[denom == 0] = 1.0
        return (vec - self.data_min) / denom

    def _denormalise_target(self, scaled_val: float) -> float:
        return float(
            scaled_val * (self.data_max[self.target_idx] - self.data_min[self.target_idx])
            + self.data_min[self.target_idx]
        )

    def _predict(self) -> Optional[float]:
        """Return predicted RPS or None if the window is not yet full."""
        if len(self.window) < self.seq_len:
            return None
        seq = np.stack(list(self.window), axis=0)   # (seq_len, n_features)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(x).item()
        return self._denormalise_target(pred_scaled)

    def step(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._step_count += 1

        # 1. Fetch current metrics
        metrics = self._fetch_metrics()
        vec = np.array([metrics.get(f, 0.0) for f in self.feature_names])
        vec = np.nan_to_num(vec, nan=0.0)
        self.window.append(self._normalise(vec))

        current = get_current_replicas(self.namespace, self.deployment)

        # 2. JVM/CPU floor: REACTIVE, needs no LSTM window. Computed FIRST so it
        # also covers the window-fill warm-up (the first ~5 min / first ramp), during
        # which the autoscaler would otherwise be blind while HPA reacts immediately.
        floor_bump = 0
        cpu_blind = False
        cpu_now = gc_pause = 0.0
        in_warmup = self._step_count <= self.warmup_grace
        if self.jvm_floor_enabled and in_warmup:
            log.info("Warm-up grace (%d/%d) — floor suppressed (JIT cold-start)",
                     self._step_count, self.warmup_grace)
        if self.jvm_floor_enabled and not in_warmup:
            gc_raw = metrics.get("jvm_gc_avg_pause_ms", float("nan"))
            cpu_raw = metrics.get("cpu_usage_percent", float("nan"))
            # NaN-DEFENSIVE: app-exposed metrics vanish exactly when the pod is
            # overwhelmed (too busy to serve its /actuator/prometheus scrape). A
            # missing CPU reading therefore signals likely saturation, NOT idle,
            # treat it as fully saturated and scale up defensively rather than going
            # blind and under-provisioning (which previously caused the death-spiral).
            cpu_blind = not np.isfinite(cpu_raw)
            cpu_now = 100.0 if cpu_blind else float(cpu_raw)
            gc_pause = 0.0 if not np.isfinite(gc_raw) else float(gc_raw)
            floor_bump = min(
                jvm_floor_bump(gc_pause, cpu_now,
                               self.gc_high, self.gc_severe, self.cpu_high),
                self.jvm_floor_max_bump,
            )

        # 3. Forecast-based recommendation once the window is full; during warm-up,
        # fall back to floor-only reactive scaling so the first ramp is covered.
        predicted_rps = self._predict()
        if predicted_rps is None:
            predicted_rps = 0.0          # numeric for logging/CSV
            rps_replicas = current
            recommended = current
            log.info("Window filling (%d/%d) — floor-only reactive (floor=%d)",
                     len(self.window), self.seq_len, floor_bump)
        else:
            rps_replicas = rps_to_replicas(
                predicted_rps, self.rps_per_replica,
                self.min_replicas, self.max_replicas,
            )
            recommended = rps_replicas

        # 3b. Apply the floor (raises the recommendation under saturation).
        if floor_bump > 0:
            floored = min(self.max_replicas, current + floor_bump)
            if floored > recommended:
                log.info(
                    "JVM floor active (%s cpu=%.0f%% gc=%.1fms): raising "
                    "recommendation %d -> %d",
                    "METRICS BLIND ->saturated" if cpu_blind else "saturation",
                    cpu_now, gc_pause, recommended, floored,
                )
                recommended = floored

        # 4. Apply safeguards
        if recommended > current:
            recommended = min(recommended, current + self.max_step)
            self.cycles_since_scaleup = 0
            action = "scale-up"
        elif recommended < current:
            self.cycles_since_scaleup += 1
            if self.cycles_since_scaleup < self.scale_down_delay:
                log.info(
                    "Scale-down suppressed (%d/%d stabilisation cycles) — "
                    "predicted_rps=%.1f, current=%d, recommended=%d",
                    self.cycles_since_scaleup, self.scale_down_delay,
                    predicted_rps, current, recommended,
                )
                recommended = current
                action = "hold"
            else:
                recommended = max(recommended, current - self.max_step)
                action = "scale-down"
        else:
            # recommended == current: load is adequately served, so RESET the
            # scale-down timer. (Previously this counted toward scale-down, so after
            # holding at N for a few minutes it would scale down even during a short
            # low phase -> re-ramp lag and SLA spikes on the next burst.) Now the
            # delay counts only CONSECUTIVE want-to-scale-down cycles, so it holds
            # through short low phases and scales down only in sustained low load.
            self.cycles_since_scaleup = 0
            action = "hold"

        # 5. Apply scaling decision
        if recommended != current:
            log.info(
                "%s: %d -> %d replicas (predicted_rps=%.1f)",
                action, current, recommended, predicted_rps,
            )
            set_replicas(self.namespace, self.deployment, recommended)
        else:
            log.debug(
                "No change: %d replicas, predicted_rps=%.1f", current, predicted_rps
            )

        # 6. Append to decision log
        row = {
            "timestamp": ts,
            "current_replicas": current,
            "predicted_rps": round(predicted_rps, 3),
            "rps_replicas": rps_replicas,
            "jvm_floor_bump": floor_bump,
            "recommended_replicas": recommended,
            "applied_replicas": recommended,
            "action": action,
        }
        row.update({k: round(float(metrics.get(k, 0)), 4) for k in self.feature_names})
        self._csv.writerow(row)
        self._log_file.flush()

    def run(self) -> None:
        log.info(
            "Predictive autoscaler started — interval: %ds, deployment: %s/%s, "
            "replicas: [%d, %d]",
            self.control_interval, self.namespace, self.deployment,
            self.min_replicas, self.max_replicas,
        )
        try:
            while True:
                t0 = time.monotonic()
                try:
                    self.step()
                except Exception as exc:
                    log.warning("Step error (will retry next cycle): %s", exc)
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, self.control_interval - elapsed))
        finally:
            self._log_file.close()
            log.info("Autoscaler stopped.")


def main():
    parser = argparse.ArgumentParser(description="JVM-Aware Predictive Autoscaler")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                        help="Prometheus base URL (default: port-forwarded localhost:9090)")
    parser.add_argument("--model", default="models/lstm_forecaster.pt")
    parser.add_argument("--scaler", default="data/processed/scaler.json")
    parser.add_argument("--namespace", default="thesis")
    parser.add_argument("--deployment", default="spring-boot-app")
    # rps-per-replica is HARDWARE-specific (calibrate per environment). Precedence:
    # CLI flag > RPS_PER_REPLICA env var > config.RPS_PER_REPLICA (the local default).
    # This lets EKS set its own calibrated value via the env var (k8s/eks/autoscaler.yaml)
    # without editing config.py, which stays the Minikube/local value.
    parser.add_argument("--rps-per-replica", type=float,
                        default=float(os.environ.get("RPS_PER_REPLICA", RPS_PER_REPLICA)),
                        help="Sustainable RPS per pod (env RPS_PER_REPLICA or config default "
                             f"{RPS_PER_REPLICA})")
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument("--control-interval", type=int, default=30,
                        help="Seconds between control-loop iterations")
    parser.add_argument("--max-step-change", type=int, default=3,
                        help="Max replicas to add/remove per cycle")
    parser.add_argument("--scale-down-delay-cycles", type=int, default=10,
                        help="Idle cycles required before scaling down (stabilisation)")
    parser.add_argument("--log-file", default="logs/predictive_autoscaler.csv")

    # JVM-pressure floor (the JVM-aware arm). Use --no-jvm-floor for the RPS-only
    # ablation. Tune thresholds with ml/backtest_jvm_floor.py before live runs.
    parser.add_argument("--jvm-floor", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("JVM_FLOOR", "on").lower()
                                not in ("0", "off", "false", "no"),
                        help="Enable the JVM-pressure scale-up floor (default: on, or env "
                             "JVM_FLOOR=off). Use --no-jvm-floor / JVM_FLOOR=off for the "
                             "RPS-only ablation arm.")
    parser.add_argument("--gc-high", type=float, default=JVM_FLOOR_GC_HIGH_MS,
                        help="Avg GC pause (ms) that adds +1 replica")
    parser.add_argument("--gc-severe", type=float, default=JVM_FLOOR_GC_SEVERE_MS,
                        help="Avg GC pause (ms) that adds +2 replicas")
    parser.add_argument("--cpu-high", type=float, default=JVM_FLOOR_CPU_HIGH_PCT,
                        help="Process CPU %% that adds +1 replica")
    parser.add_argument("--jvm-floor-max-bump", type=int, default=JVM_FLOOR_MAX_BUMP,
                        help="Max replicas the floor may add per cycle (conservative=1)")
    # Hardware/env-specific (like rps-per-replica): env WARMUP_GRACE_CYCLES > config.
    parser.add_argument("--warmup-grace-cycles", type=int,
                        default=int(os.environ.get("WARMUP_GRACE_CYCLES", WARMUP_GRACE_CYCLES)),
                        help="Suppress the JVM floor for the first N cycles (JIT cold-start). "
                             f"env WARMUP_GRACE_CYCLES or config default {WARMUP_GRACE_CYCLES}")
    args = parser.parse_args()

    PredictiveAutoscaler(args).run()


if __name__ == "__main__":
    main()
