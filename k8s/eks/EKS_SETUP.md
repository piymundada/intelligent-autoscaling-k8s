# EKS Auto Mode Setup (AWS Academy Learner Lab)

Cloud reproduction of the Minikube experiment on EKS Auto Mode, so pod scaling
becomes **real EC2 cost** (Auto Mode adds/removes nodes via Karpenter).

> **Just want the steps?** See [RUNBOOK.md](RUNBOOK.md) — a linear, copy-paste
> guide from creating clusters to pulling the data. This file explains the *why*.

> **Academy lab notes:** use the provided `LabRole` as the cluster role if the
> console lets you. Sessions are time-boxed and budget-capped (~$50-100) — EKS
> control plane is **$0.10/hr** + EC2 + ~12% Auto Mode fee. **Delete the cluster
> between sessions** (Auto Mode re-provisions nodes in ~1 min next time).

---

## Quick start (scripted)
Once the cluster exists (step 1) and `aws`/`docker`/`kubectl`/`mvn` are on PATH:
```bash
# All scripts read scripts/aws/env.sh — override defaults inline if needed:
export REGION=us-east-1 CLUSTER_NAME=<your-cluster>

./scripts/aws/build_push.sh          # build+push app + autoscaler images to ECR
./scripts/aws/deploy.sh base         # bring up app + observability + loadgen
./scripts/aws/deploy.sh hpa          # ...then run the HPA baseline
# or
./scripts/aws/deploy.sh pred         # ...the in-cluster predictive autoscaler
./scripts/aws/teardown.sh            # remove workloads between sessions (budget!)
```
The sections below explain what those scripts automate and the manual equivalents.

### Two clusters in parallel (halve wall-clock)
Run the HPA arm and the predictive arm on **separate clusters at the same time**,
so each HPA-vs-LSTM pair takes one run's wall-clock instead of two. Each cluster
gets its own kubeconfig (`env.sh` keys it off `CLUSTER_NAME`), so host-side
`kubectl` for the two never races; the two Prometheus port-forwards use ports
9090/9091.
```bash
# one-time per session (create both clusters first — step 1 — then):
for C in thesis-hpa thesis-pred; do CLUSTER_NAME=$C ./scripts/aws/build_push.sh; done
CLUSTER_NAME=thesis-hpa  ./scripts/aws/deploy.sh hpa
CLUSTER_NAME=thesis-pred ./scripts/aws/deploy.sh pred

# then each scenario in parallel (HPA on thesis-hpa, predictive on thesis-pred):
./scripts/aws/run_parallel.sh spike 20 spike_run1 60
#   -> data/spike_run1_hpa_merged.csv  +  data/spike_run1_pred_merged.csv

# teardown both at session end:
for C in thesis-hpa thesis-pred; do CLUSTER_NAME=$C ./scripts/aws/teardown.sh; done
```
A single arm on one cluster: `CLUSTER_NAME=thesis-hpa ./scripts/aws/run_eks.sh spike hpa 20 spike_run1 60`.

---

## 1. Create the EKS Auto Mode cluster
Console → EKS → Create cluster → **Auto Mode: Enabled** → cluster role `LabRole`
(or the EKS role if present) → default VPC/subnets. ~10-15 min to become Active.

Point kubectl at it:
```bash
aws eks update-kubeconfig --name <cluster-name> --region <region>
kubectl get nodes        # Auto Mode shows nodes once pods need them
```

---

## 2. Push the images to ECR  (`scripts/aws/build_push.sh`)
Minikube used `minikube image load`; EKS must pull from a registry. The script
builds **both** the app and the in-cluster predictive-autoscaler images (amd64)
and pushes them, creating the ECR repos if missing. Manual equivalent for the app:
```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=<region>                       # e.g. us-east-1
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/spring-workload-simulator

aws ecr create-repository --repository-name spring-workload-simulator --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# build (amd64 — EKS nodes are x86) and push
cd spring-workload-simulator
docker build --platform linux/amd64 -t $REPO:latest .
docker push $REPO:latest
cd ..
```

> The loadgen image (`locustio/locust`) is public — no ECR needed.

---

## 3. Apply manifests (image swapped to ECR)  (`scripts/aws/deploy.sh`)
`deploy.sh base` runs everything below (and `set image` to the ECR app image);
`deploy.sh hpa` / `deploy.sh pred` then add the chosen autoscaler. Manual steps:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/eks/nodepool.yaml          # Auto Mode NodePool (non-burstable, small nodes)

# app deployment with the ECR image (overrides the Minikube image:)
kubectl apply -f k8s/deployment.yaml
kubectl -n thesis set image deployment/spring-boot-app spring-boot-app=$REPO:latest
kubectl -n thesis patch deployment spring-boot-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"spring-boot-app","imagePullPolicy":"Always"}]}}}}'

kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/observability/prometheus-configmap.yaml
kubectl apply -f k8s/observability/prometheus-deployment.yaml
kubectl apply -f k8s/observability/grafana-deployment.yaml
kubectl apply -f k8s/hpa.yaml

# in-cluster load generator (same as Minikube)
kubectl -n thesis create configmap locust-scripts --from-file=load-testing/locust/ \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/loadgen.yaml
```

Prometheus access (port-forward works fine for *reading* metrics on EKS):
```bash
kubectl -n thesis port-forward deployment/prometheus 9090:9090 &
```

> Load is generated **in-cluster** (loadgen → ClusterIP), identical to Minikube —
> no host-side port-forward LB issue on any platform.

---

## 3b. Autoscaler: HPA vs in-cluster predictive
The two arms are mutually exclusive — run only one at a time:
- **HPA baseline:** `kubectl apply -f k8s/hpa.yaml` (or `deploy.sh hpa`).
- **Predictive (AWS-native):** `deploy.sh pred` applies `k8s/eks/autoscaler.yaml` —
  a controller Deployment (`predictive-autoscaler`) with a ServiceAccount + Role
  scoped to `get`/`scale` the app Deployment. It reads `http://prometheus:9090`
  in-cluster and bakes the trained LSTM + scaler into its image, so no host-side
  port-forward or local Python is needed. Watch it with:
  ```bash
  kubectl -n thesis logs -f deployment/predictive-autoscaler
  ```
  For the RPS-only ablation, change the `--jvm-floor` arg to `--no-jvm-floor` in
  `k8s/eks/autoscaler.yaml`.

> On Minikube the same loop still runs host-side via `scripts/run_local.sh`; the
> in-cluster Deployment is the cloud equivalent for unattended EKS runs.

---

## 4. Make the cost story materialize (pod scale -> node scale)
Auto Mode bin-packs. If all peak pods fit on ONE node, scaling pods down won't
delete a node and there's no $ gap. Ensure pods **span multiple nodes at peak**:
- The NodePool already forces **2-vCPU nodes**.
- Size pod **requests** so only ~1-2 fit per node (finalize after the calibration
  ramp — likely raise `requests.cpu` toward ~1000m, or add a
  `topologySpreadConstraints` of 1 pod/node), so 4 replicas => ~4 nodes and
  scaling to 1 => 1 node.

Verify at peak load:
```bash
kubectl get nodes                       # should be >1 node at peak
kubectl -n thesis get pods -o wide      # app pods on different nodes
```

---

## 5. Calibrate + run (same scripts as Minikube)
```bash
# one saturation ramp on the real nodes -> set RPS_PER_REPLICA in ml/config.py
LG=$(kubectl -n thesis get pod -l app=loadgen -o jsonpath='{.items[0].metadata.name}')
kubectl -n thesis exec $LG -- locust -f /mnt/locust/locustfile_ramp.py \
  --host http://spring-boot-app --headless --run-time 15m --csv /tmp/calib --csv-full-history
# then bursty/spike runs + cooldown exactly as EXPERIMENT_GUIDE.md
```

---

## 6. Measure real cost (node-hours + $)
Alongside replica-hours, record node-hours — this is where predictive's early
scale-down beats the memory-pinned HPA in actual dollars:
```bash
# node count over time (Auto Mode response to pod scaling)
kubectl get nodes -l role=experiment --no-headers | wc -l
```
Map node-hours × instance on-demand price → $. Compare HPA vs predictive over the
full run + cooldown.

---

## 7. Teardown (do this every session end — budget!)  (`scripts/aws/teardown.sh`)
```bash
./scripts/aws/teardown.sh          # deletes the thesis namespace + NodePool
# then delete the cluster (stops control-plane + node billing):
eksctl delete cluster --name <cluster-name> --region <region>   # or EKS console
```
Manual equivalent:
```bash
kubectl delete -f k8s/eks/autoscaler.yaml -f k8s/loadgen.yaml -f k8s/hpa.yaml -f k8s/eks/nodepool.yaml
```
