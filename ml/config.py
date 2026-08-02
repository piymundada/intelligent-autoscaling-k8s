"""
Shared configuration for the JVM-Aware Predictive Autoscaler project.

Single source of truth for:
  - Feature column names (used by preprocess.py, autoscaler.py)
  - Prometheus queries (used by collect_metrics.py, autoscaler.py)
  - Experiment constants (SLA threshold, scrape interval, LSTM hyperparams)
"""

# Application label used in all Prometheus queries
APP_LABEL: str = "spring-workload-simulator"

# SLA & experiment constants
SLA_THRESHOLD_MS: int = 300        # p95 response-time SLA (thesis §3.8).
                                   # NOTE: confirm against steady baseline profiling
                                   # before final runs (thesis §3.8 fixes it post-baseline).
SCRAPE_INTERVAL_S: int = 30        # Prometheus scrape / control-loop interval
WARMUP_MINUTES: int = 5            # Rows to drop at experiment start

# LSTM defaults
SEQ_LEN: int = 10                  # Lookback window (10 × 30 s = 5 min)
PRED_HORIZON: int = 2              # Steps ahead to predict (2 × 30 s = 60 s ahead).
                                   # Must exceed pod cold-start + readiness (~45-90 s)
                                   # so scaling fires BEFORE capacity is needed.
RPS_PER_REPLICA: float = 35.0      # LOCAL / Minikube calibrated value. HARDWARE-specific,
                                   # do NOT reuse on EKS. The EKS-calibrated value is set
                                   # separately via the RPS_PER_REPLICA env var in
                                   # k8s/eks/autoscaler.yaml (see scripts/aws/env.sh).
                                   # RPS per pod for the replica calc: set to the
                                   # rate that keeps per-pod CPU at HPA's 60% target
                                   # (HPA empirically used 4 pods for ~140 RPS = ~35
                                   # RPS/pod). The pod CAN sustain ~83 RPS within SLA at
                                   # steady state (in-cluster ramp), but the dynamic
                                   # workload needs this headroom so capacity is ready
                                   # before each ramp's peak: matching HPA's provisioning
                                   # so SLA is competitive, while scale-down in cooldown
                                   # delivers the cost win HPA can't (it stays pinned).

# JVM-pressure scale-up floor (thesis §3.7.2, the JVM-aware contribution)
# The floor bumps replicas when JVM/CPU pressure is high even if predicted RPS is
# low, so latency degradation from GC/heap pressure is caught (thesis §3.5.1).
# These are CONSERVATIVE defaults: they MUST be tuned per dataset with
# ml/backtest_jvm_floor.py BEFORE live runs, otherwise the floor over-fires and
# erases the cost benefit (observed in the previous experiment round).
JVM_FLOOR_GC_HIGH_MS: float = 10.0     # avg GC pause -> +1 replica
JVM_FLOOR_GC_SEVERE_MS: float = 20.0   # avg GC pause -> +2 replicas
JVM_FLOOR_CPU_HIGH_PCT: float = 60.0   # process CPU % -> +1 replica (matches HPA's
                                       # 60% target). Forecast-RPS alone under-provisions
                                       # (congestion suppresses observed RPS -> the demand
                                       # signal collapses); the CPU/GC saturation floor is
                                       # what restores adequate provisioning. Stays silent in
                                       # cooldown (CPU ~3%), preserving the scale-down cost win.
JVM_FLOOR_MAX_BUMP: int = 2            # ratchet to max in ~2 cycles under saturation

# Warm-up grace: cycles to suppress the CPU/GC floor for at controller start while
# the JVM JIT-compiles. LOCAL default 0 (4-core Minikube warmup CPU stays <60%, so
# the floor never falsely trips). On EKS the JVM sees ~1 container core, so warmup
# CPU reads 60-95% and trips the floor: set the EKS value via the
# WARMUP_GRACE_CYCLES env var in k8s/eks/autoscaler.yaml (≈3). HARDWARE-specific.
WARMUP_GRACE_CYCLES: int = 0

# Feature columns fed into the LSTM
# Only columns reliably available via Micrometer/Prometheus in this lab setup.
# memory_usage_bytes is excluded: process_resident_memory_bytes is not scraped
# by the current Prometheus config (no cAdvisor deployed).
# response_time_p95_ms is included: populated from Locust stats_history.csv
# via merge_locust_prometheus.py for training; falls back gracefully if absent.
FEATURE_COLS: list = [
    "request_rate_rps",
    "response_time_p95_ms",
    "response_time_max_ms",
    "cpu_usage_percent",
    "jvm_heap_used_bytes",
    "jvm_heap_utilisation",
    "jvm_gc_avg_pause_ms",
    "jvm_gc_pause_count",
    "jvm_threads_live",
    "replica_count",
]

TARGET_COL: str = "request_rate_rps"

# Prometheus queries
# These queries must return data from the Micrometer/Actuator metrics exposed by
# spring-workload-simulator. cAdvisor / kube-state-metrics are NOT deployed.
# All queries assume port-forwarded Prometheus at localhost:9090.
PROMETHEUS_QUERIES: dict = {
    # Workload layer
    "request_rate_rps": (
        f'sum(rate(http_server_requests_seconds_count'
        f'{{application="{APP_LABEL}"}}[1m]))'
    ),
    "response_time_max_ms": (
        f'max(http_server_requests_seconds_max'
        f'{{application="{APP_LABEL}"}}) * 1000'
    ),
    "error_rate": (
        f'sum(rate(http_server_requests_seconds_count'
        f'{{application="{APP_LABEL}",status=~"5.."}}[1m])) / '
        f'clamp_min(sum(rate(http_server_requests_seconds_count'
        f'{{application="{APP_LABEL}"}}[1m])), 1)'
    ),
    "throughput_rps": (
        f'sum(rate(http_server_requests_seconds_count'
        f'{{application="{APP_LABEL}",status=~"2.."}}[1m]))'
    ),
    # Infrastructure layer (Micrometer process metrics, no cAdvisor needed)
    # avg() across pods: returns ONE series even when scaled to multiple replicas,
    # matching what HPA targets. Without avg() the query returns one series per pod
    # and collect_metrics (which takes result[0]) would grab an arbitrary/stale pod.
    "cpu_usage_percent": (
        f'avg(process_cpu_usage{{application="{APP_LABEL}"}}) * 100'
    ),
    "system_cpu_usage_percent": (
        f'avg(system_cpu_usage{{application="{APP_LABEL}"}}) * 100'
    ),
    "replica_count": (
        'sum(up{job="spring-boot-app"})'
    ),
    # JVM layer (Micrometer / Actuator)
    "jvm_gc_pause_seconds_sum": (
        f'sum(increase(jvm_gc_pause_seconds_sum'
        f'{{application="{APP_LABEL}"}}[1m]))'
    ),
    "jvm_gc_pause_count": (
        f'sum(increase(jvm_gc_pause_seconds_count'
        f'{{application="{APP_LABEL}"}}[1m]))'
    ),
    "jvm_heap_used_bytes": (
        f'sum(jvm_memory_used_bytes'
        f'{{application="{APP_LABEL}",area="heap"}})'
    ),
    "jvm_heap_committed_bytes": (
        f'sum(jvm_memory_committed_bytes'
        f'{{application="{APP_LABEL}",area="heap"}})'
    ),
    "jvm_threads_live": (
        f'sum(jvm_threads_live_threads{{application="{APP_LABEL}"}})'
    ),
    "jvm_threads_peak": (
        f'sum(jvm_threads_peak_threads{{application="{APP_LABEL}"}})'
    ),
    "jvm_classes_loaded": (
        f'jvm_classes_loaded_classes{{application="{APP_LABEL}"}}'
    ),
    # Derived metrics: computable from Prometheus at runtime
    "jvm_heap_utilisation": (
        f'sum(jvm_memory_used_bytes{{application="{APP_LABEL}",area="heap"}}) / '
        f'sum(jvm_memory_committed_bytes{{application="{APP_LABEL}",area="heap"}})'
    ),
    "jvm_gc_avg_pause_ms": (
        f'sum(increase(jvm_gc_pause_seconds_sum{{application="{APP_LABEL}"}}[1m])) / '
        f'clamp_min(sum(increase(jvm_gc_pause_seconds_count{{application="{APP_LABEL}"}}[1m])), 1)'
        f' * 1000'
    ),
    "response_time_p95_ms": (
        f'histogram_quantile(0.95, sum(rate(http_server_requests_seconds_bucket'
        f'{{application="{APP_LABEL}"}}[1m])) by (le)) * 1000'
    ),
}

# Subset of PROMETHEUS_QUERIES used at runtime by the predictive autoscaler.
# All 10 FEATURE_COLS are now queryable directly from Prometheus.
AUTOSCALER_FEATURE_QUERIES: dict = {
    k: v for k, v in PROMETHEUS_QUERIES.items()
    if k in FEATURE_COLS
}


# JVM-pressure floor logic (single source of truth)
# Shared by the live autoscaler (ml/autoscaler.py) and the offline tuning tool
# (ml/backtest_jvm_floor.py) so live behaviour exactly matches what the backtest
# predicts. Returns the number of EXTRA replicas to add on top of the RPS-based
# recommendation, based on current JVM/CPU pressure.
def jvm_floor_bump(
    gc_pause_ms: float,
    cpu_percent: float,
    gc_high: float = JVM_FLOOR_GC_HIGH_MS,
    gc_severe: float = JVM_FLOOR_GC_SEVERE_MS,
    cpu_high: float = JVM_FLOOR_CPU_HIGH_PCT,
) -> int:
    bump = 0
    if gc_pause_ms >= gc_severe:
        bump = max(bump, 2)
    elif gc_pause_ms >= gc_high:
        bump = max(bump, 1)
    if cpu_percent >= cpu_high:
        bump = max(bump, 1)
    return bump
