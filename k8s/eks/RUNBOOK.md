# EKS Runbook — end to end

A linear, copy-paste guide: create clusters → push images → deploy → run the
HPA-vs-predictive experiment in parallel → pull the data from Prometheus →
compare → tear down. For the *why* behind each piece see [EKS_SETUP.md](EKS_SETUP.md).

Defaults used throughout: region `us-east-1`, clusters `thesis-hpa` and
`thesis-pred`. Override with `REGION` / `CLUSTER_NAME` env vars.

---

## 0. Prerequisites (once, on your laptop)

| Tool | Check | Notes |
|------|-------|-------|
| AWS CLI | `aws sts get-caller-identity` | Must print your account. In an Academy Learner Lab, paste the lab's AWS CLI credentials first. |
| kubectl | `kubectl version --client` | |
| Docker | `docker info` | Daemon must be running (build + push images). |
| Java 17 + Maven | `mvn -v` | Builds the Spring app jar. |
| Python 3.11 + deps | `python3 -c "import requests,pandas"` | For metric collection/analysis: `pip install -r requirements.txt`. |

> All `scripts/aws/*.sh` read `scripts/aws/env.sh`, which derives your ECR
> registry and sets a **per-cluster kubeconfig** (`~/.kube/<cluster>.config`)
> so the two clusters never clash. Run every command from the repo root.

---

## 1. Create the two EKS Auto Mode clusters

You need two clusters so the HPA arm and the predictive arm run in parallel,
fully isolated (clean p95 latency + node-hour attribution).

**AWS Academy Learner Lab (console — recommended there):** for **each** of
`thesis-hpa` and `thesis-pred`:
Console → EKS → **Create cluster** → **Auto Mode: Enabled** → cluster role
`LabRole` → default VPC/subnets. ~10–15 min each to become *Active*.

**Outside Academy (eksctl, if you have IAM rights):**
```bash
for C in thesis-hpa thesis-pred; do
  eksctl create cluster --name "$C" --region us-east-1 --enable-auto-mode
done
```
> Auto Mode support needs a recent eksctl — verify the flag with `eksctl create cluster --help | grep -i auto`. If your eksctl lacks it, use the console.

Point a kubeconfig at each (writes to the isolated per-cluster file):
```bash
for C in thesis-hpa thesis-pred; do
  KUBECONFIG=~/.kube/$C.config aws eks update-kubeconfig --name "$C" --region us-east-1
done
```

### 1b. metrics-server (HPA cluster — required)
The HPA scales on CPU+memory, which needs metrics-server.

> **EKS 1.35+ ships metrics-server as a MANAGED ADDON already** — do NOT
> `kubectl apply` the upstream copy (it collides on immutable fields and breaks the
> Metrics API). Just confirm the addon is present/healthy:
```bash
export KUBECONFIG=~/.kube/thesis-hpa.config
aws eks list-addons --cluster-name thesis-hpa --region us-east-1   # expect "metrics-server"
kubectl top nodes                                                  # works within ~1-2 min of cluster ready
```
If `top` errors with "Metrics API not available", force the addon back to a clean
state (fixes any prior tampering):
```bash
aws eks update-addon --cluster-name thesis-hpa --region us-east-1 \
  --addon-name metrics-server --resolve-conflicts OVERWRITE
```
If `list-addons` does NOT include metrics-server (older EKS), then — and only then —
install upstream: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`.

---

## 2. Build & push the images to ECR (once — shared by both clusters)

ECR is account/region-scoped, so build once. The script Maven-builds the app,
builds both images for `linux/amd64`, creates the repos if missing, and pushes.
```bash
./scripts/aws/build_push.sh          # app + predictive-autoscaler
```
Prereq for the autoscaler image: a trained model must exist
(`ml/models/lstm_forecaster.pt` + `data/processed/scaler.json`). If not, run the
ML pipeline first (`python ml/train.py …`).

---

## 3. Deploy — HPA on one cluster, predictive on the other

```bash
CLUSTER_NAME=thesis-hpa  ./scripts/aws/deploy.sh hpa
CLUSTER_NAME=thesis-pred ./scripts/aws/deploy.sh pred
```
Each applies: namespace, Auto Mode NodePool, the app (image swapped to ECR),
Service, Prometheus, Grafana, the in-cluster Locust loadgen, and the chosen
autoscaler (HPA `Deployment` vs the `predictive-autoscaler` controller + RBAC).
`deploy.sh` refuses to leave both autoscalers running.

---

## 4. Verify each cluster is healthy

```bash
export KUBECONFIG=~/.kube/thesis-hpa.config     # repeat for thesis-pred
kubectl -n thesis get pods                  # app, prometheus, grafana, loadgen Running
kubectl get nodes                           # Auto Mode nodes appear once pods schedule
kubectl -n thesis get hpa                   # (hpa cluster) shows targets, not <unknown>
kubectl -n thesis logs deploy/predictive-autoscaler --tail=20   # (pred cluster) "Window filling…"
```
Confirm Prometheus is scraping the app (should be non-empty):
```bash
kubectl -n thesis port-forward deployment/prometheus 9090:9090 &
curl -s 'http://localhost:9090/api/v1/query?query=up{job="spring-boot-app"}' | python3 -m json.tool
kill %1
```

---

## 5. Run the experiment (both arms in parallel)

One command runs the same scenario on both clusters at once and waits for both:
```bash
# run_parallel.sh <scenario> <observe_min> <label_prefix> [load_min]
./scripts/aws/run_parallel.sh spike 180 spike_run1 60
```
Scenarios: `steady | bursty | spike | gc`. It does, per arm: reset to a fresh
1-replica pod → drive in-cluster Locust load for `load_min` → **observe for the
full `observe_min`** without resetting, recording if/when the app scales back to
the 1-replica baseline → collect over the whole window.

> **Observation window (`observe_min`):** use **180 (3 h)** for the headline run.
> The app is *not* rescaled during it — we just watch. This is what surfaces the
> thesis finding: HPA's committed JVM heap keeps memory utilisation high, so it
> stays **pinned** and never scales in, while the predictive arm returns to 1
> replica quickly. The run log records `returned_to_1=yes/no … final_replicas=N`
> for each arm. (HPA's floor is `minReplicas: 1`, so 1 — not 0 — is the baseline.)

Watch progress live:
```bash
tail -f /tmp/run_spike_run1_hpa.txt /tmp/run_spike_run1_pred.txt
```
Outputs when done:
- `data/spike_run1_hpa_merged.csv`   — HPA metrics (Prometheus ⨝ Locust)
- `data/spike_run1_pred_merged.csv`  — predictive metrics
- `logs/spike_run1_pred_autoscaler.csv` — the predictive controller's decision log

> Single arm instead of parallel:
> `CLUSTER_NAME=thesis-hpa ./scripts/aws/run_eks.sh spike hpa 20 spike_run1 60`

---

## 6. Get the data from Prometheus

**Automatic:** step 5 already pulls it — `run_eks.sh` port-forwards each cluster's
Prometheus (HPA on `:9090`, predictive on `:9091`) and runs
`ml/collect_metrics.py` for the run's UTC window, then merges with the Locust CSV.

**Manual / ad-hoc** (inspect or re-pull a window):
```bash
export KUBECONFIG=~/.kube/thesis-pred.config
kubectl -n thesis port-forward deployment/prometheus 9091:9090 &

# explore in the UI:           open http://localhost:9091
# one PromQL query:
curl -s 'http://localhost:9091/api/v1/query?query=sum(rate(http_server_requests_seconds_count{application="spring-workload-simulator"}[1m]))'

# re-export a full window to CSV (all feature metrics):
PYTHONPATH=ml python3 ml/collect_metrics.py \
  --prometheus-url http://localhost:9091 \
  --start 2026-06-13T10:00:00Z --end 2026-06-13T11:30:00Z \
  --output data/manual_pull.csv
kill %1
```
Grafana dashboards (optional): `kubectl -n thesis port-forward deployment/grafana 3000:3000` → http://localhost:3000.

---

## 7. Compare HPA vs predictive

```bash
python analysis/compare.py \
  --hpa-csv       data/spike_run1_hpa_merged.csv \
  --pred-metrics  data/spike_run1_pred_merged.csv \
  --pred-csv      logs/spike_run1_pred_autoscaler.csv \
  --model-results ml/models/lstm_forecaster_results.json \
  --output-dir    results/spike/
```

**Real $ cost (node-hours):** with separate clusters, node-hours are clean per arm.
Sample during the run (or scrape Auto Mode events afterward):
```bash
KUBECONFIG=~/.kube/thesis-hpa.config  kubectl get nodes -l role=experiment --no-headers | wc -l
KUBECONFIG=~/.kube/thesis-pred.config kubectl get nodes -l role=experiment --no-headers | wc -l
```
Map node-hours × the instance on-demand price → $, and compare.

Repeat steps 5–7 for each scenario/run (`bursty`, `gc`, more `run` numbers).

---

## Cost guard (run anytime)

Check what's billing and the current burn rate — and get a ⚠️ if clusters/nodes
are up with no active run:
```bash
./scripts/aws/status.sh
```
Run it before logging off; the single biggest cost mistake is leaving clusters up.

---

## 8. Teardown (every session end — budget!)

```bash
for C in thesis-hpa thesis-pred; do CLUSTER_NAME=$C ./scripts/aws/teardown.sh; done
# then delete BOTH clusters to stop control-plane billing:
for C in thesis-hpa thesis-pred; do eksctl delete cluster --name "$C" --region us-east-1; done
# (Academy: delete each cluster in the EKS console instead.)
```
