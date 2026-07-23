# Experiment Runbook

Step-by-step guide to reproduce the thesis experiments on a local Minikube cluster.

## Prerequisites

| Tool | Version |
|---|---|
| Minikube | >= 1.33 |
| kubectl | >= 1.29 |
| Docker Desktop | >= 25 (allocate at least 5 GB RAM to Docker) |
| Python | 3.11+ |
| Maven | 3.9+ |
| Locust | >= 2.24 |

---

## 1. Start Minikube

```bash
minikube start --cpus=4 --memory=4500
```

> Note: `--memory=6g` will fail if Docker Desktop has less than 6 GB allocated.
> 4500 MB is the safe default for a typical 8 GB machine.

---

## 2. Build and Deploy the Spring Boot App

```bash
cd spring-workload-simulator
mvn clean package -DskipTests

docker build -t spring-workload-simulator:latest .

# Load into Minikube (no registry needed)
minikube image load spring-workload-simulator:latest

# Deploy namespace, app, and service
kubectl apply -f ../k8s/namespace.yaml
kubectl apply -f ../k8s/deployment.yaml
kubectl apply -f ../k8s/service.yaml

# Verify app is up
kubectl -n thesis get pods
# Wait until READY 1/1
```

Confirm metrics endpoint is working:
```bash
kubectl -n thesis port-forward service/spring-boot-app-nodeport 8080:8080 &
curl http://localhost:8080/actuator/prometheus | head -20
```

---

## 3. Deploy Observability Stack

```bash
kubectl apply -f k8s/observability/prometheus-configmap.yaml
kubectl apply -f k8s/observability/prometheus-deployment.yaml
kubectl apply -f k8s/observability/grafana-deployment.yaml

# Wait for pods to be ready
kubectl -n thesis get pods

# Port-forward Prometheus
kubectl -n thesis port-forward deployment/prometheus 9090:9090 &

# Prometheus UI
open http://localhost:9090

# Grafana UI (admin / admin)
kubectl -n thesis port-forward deployment/grafana 3000:3000 &
open http://localhost:3000
```

---

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** invoke Locust as `python3 -m locust …` (used throughout this guide).
> The bare `locust` console-script is often not on PATH and fails silently as
> `command not found`, which would waste a 30-min run. The commands below already
> use the module form.

---

## 5. Run Baseline HPA Experiments

Apply the HPA:
```bash
kubectl apply -f k8s/hpa.yaml
```

Ensure port-forwards are running (app on :8080, Prometheus on :9090), then run each
load scenario. Note the **UTC** start/end times — required for Prometheus collection.

> The HPA uses **CPU + memory** (`k8s/hpa.yaml`), per proposal §8.6.1 and real-world
> practice. Because the JVM holds committed heap, memory stays high (~66% at idle)
> and the HPA tends to scale up on memory and resist scaling in — a documented
> *finding* showing why infra metrics poorly capture JVM workloads. See the comment
> in `k8s/hpa.yaml`.

> These baseline runs serve **two purposes**: they are the HPA reference *and* the
> LSTM training data. Run each pattern a few times so the model sees the ramp
> shapes (steady ×3, bursty ×2, spike ×2 is a good minimum).

```bash
# Steady load (30 min, 20 users) — run 3 times
python3 -m locust -f load-testing/locust/locustfile_steady.py \
       --host=http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 30m --headless \
       --csv=data/hpa_steady_run1 --csv-full-history

# Bursty load (30 min, bursts up to 30 users) — run 2 times
python3 -m locust -f load-testing/locust/locustfile_bursty.py \
       --host=http://localhost:8080 \
       --users 30 --spawn-rate 5 --run-time 30m --headless \
       --csv=data/hpa_bursty_run1 --csv-full-history

# Spike load (30 min, spikes up to 40 users) — run 2 times
python3 -m locust -f load-testing/locust/locustfile_spike.py \
       --host=http://localhost:8080 \
       --users 40 --spawn-rate 20 --run-time 30m --headless \
       --csv=data/hpa_spike_run1 --csv-full-history
```

After each run, collect Prometheus metrics (replace UTC timestamps):
```bash
python ml/collect_metrics.py \
  --prometheus-url http://localhost:9090 \
  --start "2024-01-01T10:00:00Z" \
  --end   "2024-01-01T10:30:00Z" \
  --output data/hpa_steady_run1.csv
```

Merge Locust p95 data with Prometheus metrics:
```bash
python ml/merge_locust_prometheus.py \
  --locust-csv  data/hpa_steady_run1_stats_history.csv \
  --prom-csv    data/hpa_steady_run1.csv \
  --output      data/hpa_steady_run1_merged.csv
```

Collect/merge every baseline run (steady ×3, bursty ×2, spike ×2). All of these
merged CSVs are used as LSTM training data in Step 6.

---

## 5b. Calibrate Capacity and SLA Target

Before modelling, derive `RPS_PER_REPLICA` from a steady baseline instead of
guessing — it drives every predictive scaling decision. This also confirms the
300 ms SLA target is realistic (paper §8.7 fixes the SLA after baseline profiling).

```bash
PYTHONPATH=ml python ml/calibrate_capacity.py \
  --input data/hpa_steady_run1_merged.csv
```

Set `RPS_PER_REPLICA` in `ml/config.py` to the reported value (or pass
`--rps-per-replica <value>` to the autoscaler in Step 7). If the script warns that
median p95 already exceeds the SLA, raise the threshold or the baseline replicas.

> **Caveat:** the steady scenario only offers ~20 RPS to a 1-replica pod that sits
> at ~9% CPU / ~12 ms p95 — it never reaches its knee, so the reported value is a
> *lower bound* (the pod can clearly do more). For a true per-replica capacity,
> run a one-off **saturation ramp** (HPA deleted, `--replicas 1` pinned, users
> increasing) until p95 approaches 300 ms, and read RPS/replica at that point.

---

## 6. Train the LSTM Model

Preprocess **all** baseline runs (steady + bursty + spike) into sequence windows.
Including the bursty/spike ramps is essential: a model trained only on steady load
has never seen a rising edge and cannot anticipate spikes — it would merely react.
The live evaluation runs in Step 7 are fresh runs, so this is not leakage.

```bash
python ml/preprocess.py \
  --inputs data/hpa_steady_run1_merged.csv \
           data/hpa_steady_run2_merged.csv \
           data/hpa_steady_run3_merged.csv \
           data/hpa_bursty_run1_merged.csv \
           data/hpa_bursty_run2_merged.csv \
           data/hpa_spike_run1_merged.csv \
           data/hpa_spike_run2_merged.csv \
  --output-dir data/processed/ \
  --seq-len 10 \
  --pred-horizon 2 \
  --target request_rate_rps \
  --warmup-minutes 0
```

> `--pred-horizon 2` = 60 s ahead (2 × 30 s), chosen to exceed pod cold-start +
> readiness so scaling fires *before* capacity is needed. Keep it in
> sync with `PRED_HORIZON` in `ml/config.py` (currently `2`).
>
> `--warmup-minutes 0` is required: `merge_locust_prometheus.py` already strips the
> 5-min warm-up. Without this flag the warm-up is dropped twice (10 min lost/run).

Train the LSTM:
```bash
python ml/train.py \
  --data-dir   data/processed/ \
  --model-out  ml/models/lstm_forecaster.pt \
  --epochs 100 --hidden-size 16 --num-layers 1
```

> The **local** model is compact: **hidden 16, 1 layer** (early-stops ~epoch 67). The
> **EKS** model is retrained separately and is larger (**hidden 64, 2 layers**) — see
> [k8s/eks/RUNBOOK.md](k8s/eks/RUNBOOK.md).

---

## 6b. Tune the JVM-Pressure Floor (offline gate — do this before live runs)

The JVM floor is what makes the autoscaler *JVM-aware*: it adds a replica when GC
pause / CPU pressure is high even if predicted RPS is low. **Risk:** if thresholds
are too low it fires constantly and erases the cost benefit (observed previously).
Tune the thresholds against the baseline data **before** spending a 30-min live run:

```bash
PYTHONPATH=ml python ml/backtest_jvm_floor.py \
  --glob "data/hpa_bursty_*_merged.csv" \
  --gc-high 10 --gc-severe 20 --cpu-high 70
```

Read the summary:
- **Breach coverage** high (floor fires when SLA breaches happen) — good.
- **Cost increase %** low, **false-positive %** low — good.

Raise `--gc-high` / `--cpu-high` until cost increase and false positives drop to an
acceptable level, then put the chosen values in `ml/config.py`
(`JVM_FLOOR_GC_HIGH_MS`, `JVM_FLOOR_CPU_HIGH_PCT`) or pass them as flags in Step 7.

---

## 7. Run Predictive Autoscaler Experiments

> Evaluation runs are **fresh** bursty and spike runs under the predictive
> autoscaler, compared against the HPA baseline runs from Step 5.
>
> Run **two arms** per scenario to isolate the JVM-aware contribution:
>   1. **JVM-aware** (default): RPS forecast + JVM floor.
>   2. **RPS-only ablation**: add `--no-jvm-floor`.
>
> Comparing the two shows exactly what the JVM floor costs and what SLA benefit it
> buys — directly addressing the cost-vs-benefit tuning concern.

**Recommended: use the runbook script** — it handles the fresh-pod reset, the
in-cluster loadgen, collection and merging in one shot (scenarios: `steady`,
`bursty`, `spike`, `gc`; modes: `hpa`, `pred`, `pred-rpsonly`):
```bash
# JVM-aware arm, then the RPS-only ablation arm (60-min load, 20-min cooldown cap)
bash scripts/run_local.sh bursty pred         20 pred_bursty_run1         60
bash scripts/run_local.sh bursty pred-rpsonly 20 pred_bursty_rpsonly_run1 60
bash scripts/run_local.sh spike  pred-rpsonly 20 pred_spike_rpsonly_run1  60
bash scripts/run_local.sh gc     pred         20 pred_gc_run1             60
```
Requires a live Prometheus port-forward (`kubectl -n thesis port-forward
svc/prometheus 9090:9090`). The manual per-terminal flow below is equivalent.

Disable HPA so it does not conflict:
```bash
kubectl -n thesis scale deployment/spring-boot-app --replicas=1
kubectl delete hpa spring-boot-hpa -n thesis
```

### Bursty run

Start the autoscaler in one terminal (note `PYTHONPATH=ml`). JVM floor is on by
default; thresholds come from `ml/config.py` (override with `--gc-high` etc.):
```bash
# Arm 1: JVM-aware (default)
PYTHONPATH=ml python3.11 ml/autoscaler.py \
  --prometheus-url http://localhost:9090 \
  --model      ml/models/lstm_forecaster.pt \
  --scaler     data/processed/scaler.json \
  --max-replicas 4 \
  --log-file   logs/pred_bursty_run1.csv

# Arm 2: RPS-only ablation (add --no-jvm-floor, log to a separate file)
PYTHONPATH=ml python3.11 ml/autoscaler.py \
  --prometheus-url http://localhost:9090 \
  --model      ml/models/lstm_forecaster.pt \
  --scaler     data/processed/scaler.json \
  --max-replicas 4 --no-jvm-floor \
  --log-file   logs/pred_bursty_run1_rpsonly.csv
```

In a second terminal, run the bursty load:
```bash
python3 -m locust -f load-testing/locust/locustfile_bursty.py \
       --host=http://localhost:8080 \
       --users 30 --spawn-rate 5 --run-time 30m --headless \
       --csv=data/pred_bursty_run1 --csv-full-history
```

After Locust finishes, stop the autoscaler (`Ctrl+C`), then collect and merge:
```bash
python ml/collect_metrics.py \
  --prometheus-url http://localhost:9090 \
  --start "2024-01-01T10:00:00Z" \
  --end   "2024-01-01T10:30:00Z" \
  --output data/pred_bursty_run1.csv

python ml/merge_locust_prometheus.py \
  --locust-csv  data/pred_bursty_run1_stats_history.csv \
  --prom-csv    data/pred_bursty_run1.csv \
  --output      data/pred_bursty_run1_merged.csv
```

### Spike run

Repeat the same steps with the spike locustfile:
```bash
# Reset to 1 replica first
kubectl -n thesis scale deployment/spring-boot-app --replicas=1

PYTHONPATH=ml python3.11 ml/autoscaler.py \
  --prometheus-url http://localhost:9090 \
  --model      ml/models/lstm_forecaster.pt \
  --scaler     data/processed/scaler.json \
  --max-replicas 4 \
  --log-file   logs/pred_spike_run1.csv
```

```bash
python3 -m locust -f load-testing/locust/locustfile_spike.py \
       --host=http://localhost:8080 \
       --users 40 --spawn-rate 20 --run-time 30m --headless \
       --csv=data/pred_spike_run1 --csv-full-history
```

```bash
python ml/collect_metrics.py \
  --prometheus-url http://localhost:9090 \
  --start "2024-01-01T10:00:00Z" \
  --end   "2024-01-01T10:30:00Z" \
  --output data/pred_spike_run1.csv

python ml/merge_locust_prometheus.py \
  --locust-csv  data/pred_spike_run1_stats_history.csv \
  --prom-csv    data/pred_spike_run1.csv \
  --output      data/pred_spike_run1_merged.csv
```

---

## 8. Analyse and Generate Figures

```bash
# Bursty comparison
python analysis/compare.py \
  --hpa-csv      data/hpa_bursty_run1_merged.csv \
  --pred-metrics data/pred_bursty_run1_merged.csv \
  --pred-csv     logs/pred_bursty_run1.csv \
  --model-results ml/models/lstm_forecaster_results.json \
  --output-dir   results/bursty/

# Spike comparison
python analysis/compare.py \
  --hpa-csv      data/hpa_spike_run1_merged.csv \
  --pred-metrics data/pred_spike_run1_merged.csv \
  --pred-csv     logs/pred_spike_run1.csv \
  --model-results ml/models/lstm_forecaster_results.json \
  --output-dir   results/spike/

# Figures (run for each scenario)
python analysis/visualise.py \
  --hpa-metrics  data/hpa_bursty_run1_merged.csv \
  --pred-metrics data/pred_bursty_run1_merged.csv \
  --model-results ml/models/lstm_forecaster_results.json \
  --output-dir   results/figures/
```

> Note: `--test-preds` and `--test-targets` (`.npy` files) are git-ignored.
> Re-run `ml/train.py` to regenerate them before running `visualise.py`.

Figures produced: replica timelines, latency comparisons, SLA violation rates,
resource utilisation, JVM metrics, LSTM training curves, predicted vs actual RPS.
