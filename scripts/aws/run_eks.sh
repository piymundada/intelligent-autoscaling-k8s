#!/usr/bin/env bash
# One experiment run on an EKS cluster: fresh 1-replica start -> load -> observe
# -> collect/merge. The EKS analogue of scripts/run_local.sh, with two differences:
#   - Prometheus is reached via a port-forward on a CONFIGURABLE local port (so two
#     clusters can be scraped in parallel without colliding on 9090).
#   - The autoscaler under test runs IN-CLUSTER (HPA controller, or the
#     predictive-autoscaler Deployment): the host never runs the Python loop.
#
# Targets the cluster named by CLUSTER_NAME (via the per-cluster kubeconfig set in
# env.sh), so run two of these in parallel with different CLUSTER_NAME + ports.
#
# Usage: CLUSTER_NAME=thesis-hpa run_eks.sh <scenario> <mode> <observe_min> <label> [load_min] [prom_port]
#   scenario   : steady | bursty | spike | gc
#   mode       : hpa | pred           (must already be deployed via deploy.sh <mode>)
#   observe_min: post-load OBSERVATION window in minutes (0 = none). The app is NOT
#                reset/rescaled during it: we watch for the FULL duration and record
#                IF/WHEN it scaled back to the baseline (1 replica). For the HPA
#                pinned-cost finding use 180 (3h): committed JVM heap keeps memory
#                high so HPA never scales in. (NB: minReplicas=1, so 1 is the floor,
#                it cannot reach 0.)
#   label      : output prefix, e.g. hpa_spike_run1
#   load_min   : load duration minutes (default 30)
#   prom_port  : local port for the Prometheus port-forward (default 9090)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
# env.sh enables `set -euo pipefail` (right for build/deploy). This run harness is
# best-effort by design: kill of a dead port-forward, racing kubectl cp, grep with
# no match are all expected to return non-zero without aborting. Restore run_local's
# behaviour: keep -u, drop -e/pipefail.
set +e +o pipefail
cd "$ROOT"

SCEN=$1; MODE=$2; COOL=$3; LABEL=$4; LOAD_MIN=${5:-30}; PROM_PORT=${6:-9090}
NS="$NAMESPACE"
ASC=predictive-autoscaler        # in-cluster autoscaler Deployment name
RUN=/tmp/run_${LABEL}.txt; : > "$RUN"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN"; }

# Host python for collect/merge (needs pandas/numpy/requests). Homebrew's python3
#: pulled in by awscli, lacks them; /usr/bin/python3 (system) has them. Override
# with PY=... if needed.
if [ -z "${PY:-}" ]; then
  for c in /usr/bin/python3 python3 python3.11 python3.9; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c "import pandas,requests" >/dev/null 2>&1 && { PY="$c"; break; }
  done
fi
[ -z "${PY:-}" ] && { echo "ERROR: no python with pandas/requests (pip install -r requirements.txt)"; exit 1; }

log "RUN $LABEL ($SCEN/$MODE cluster=$CLUSTER_NAME observe=${COOL}m prom_port=$PROM_PORT py=$PY)"

aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION" >/dev/null

# Prometheus port-forward for metric collection (distinct port per parallel run).
kubectl -n "$NS" port-forward deployment/prometheus "${PROM_PORT}:9090" >/tmp/${LABEL}_pf.log 2>&1 &
PF_PID=$!
cleanup(){ kill "$PF_PID" 2>/dev/null; }
trap cleanup EXIT
sleep 5
PROM="http://localhost:${PROM_PORT}"

# 1. Stop the predictive autoscaler during reset so it can't fight the scale-to-0
#    (no pods -> blind metrics -> it would otherwise scale UP defensively).
if [ "$MODE" = "pred" ]; then
  if ! kubectl -n "$NS" get deploy "$ASC" >/dev/null 2>&1; then
    log "ERROR: $ASC not deployed — run scripts/aws/deploy.sh pred first"; exit 1
  fi
  kubectl -n "$NS" scale deploy/"$ASC" --replicas=0 >/dev/null 2>&1
fi

# 2. GUARANTEE a fresh, single-replica start (same logic as run_local.sh).
kubectl -n "$NS" delete hpa spring-boot-hpa --ignore-not-found >/dev/null 2>&1
kubectl -n "$NS" scale deploy/spring-boot-app --replicas=0 >/dev/null 2>&1
t=0; until [ "$(kubectl -n "$NS" get pods -l app=spring-boot-app --no-headers 2>/dev/null | wc -l | tr -d ' ')" = "0" ] || [ $t -ge 36 ]; do sleep 5; t=$((t+1)); done
kubectl -n "$NS" scale deploy/spring-boot-app --replicas=1 >/dev/null 2>&1
kubectl -n "$NS" rollout status deploy/spring-boot-app --timeout=180s >/dev/null 2>&1
t=0; until [ "$(kubectl -n "$NS" get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')" = "1" ] || [ $t -ge 36 ]; do sleep 5; t=$((t+1)); done
sleep 12
NP=$(kubectl -n "$NS" get pods -l app=spring-boot-app --no-headers 2>/dev/null | wc -l | tr -d ' ')
log "fresh start: pods=$NP (should be 1)"

# 3. Enable the autoscaler under test.
if [ "$MODE" = "hpa" ]; then
  kubectl apply -f k8s/eks/hpa.yaml >/dev/null 2>&1
elif [ "$MODE" = "pred" ]; then
  kubectl -n "$NS" scale deploy/"$ASC" --replicas=1 >/dev/null 2>&1
  kubectl -n "$NS" rollout status deploy/"$ASC" --timeout=120s >/dev/null 2>&1
fi

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"
log "load start"

# 4. In-cluster load.
kubectl -n "$NS" rollout status deploy/loadgen --timeout=120s >/dev/null 2>&1
LG=$(kubectl -n "$NS" get pod -l app=loadgen --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
case "$SCEN" in
  steady) USERS="--users 20 --spawn-rate 2" ;;
  bursty) USERS="--users 180 --spawn-rate 40" ;;
  spike)  USERS="--users 180 --spawn-rate 60" ;;
  gc)     USERS="--users 80 --spawn-rate 20" ;;
  *)      USERS="--users 20 --spawn-rate 2" ;;
esac
kubectl -n "$NS" exec "$LG" -- sh -c "rm -f /tmp/${LABEL}*" 2>/dev/null
kubectl -n "$NS" exec "$LG" -- sh -c "LOAD_SECONDS=$((LOAD_MIN*60)) locust -f /mnt/locust/locustfile_${SCEN}.py \
  --host http://spring-boot-app --headless --run-time ${LOAD_MIN}m $USERS \
  --csv /tmp/${LABEL} --csv-full-history" >/tmp/${LABEL}_locust.log 2>&1
echo "LOAD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"
log "load done"

# 5. Observation window: watch for the FULL observe_min (no early exit, no reset),
#    recording IF/WHEN the app scaled back to baseline (1 replica). For HPA this
#    captures the pinned-cost finding (committed JVM heap keeps it from scaling in);
#    for predictive it records how fast it recovered while still logging the whole
#    same-horizon window so replica-hours are comparable across arms.
if [ "$COOL" -gt 0 ]; then
  c=0; CAP=$((COOL*4)); first_at=""
  while [ $c -lt $CAP ]; do
    R=$(kubectl -n "$NS" get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')
    [ -z "$first_at" ] && [ "$R" = "1" ] && first_at=$((c*15))
    [ $((c % 20)) -eq 0 ] && log "observe t=$((c*15))s replicas=${R:-0}"
    sleep 15; c=$((c+1))
  done
  FINAL=$(kubectl -n "$NS" get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')
  if [ -n "$first_at" ]; then
    echo "returned_to_1=yes  first_at=${first_at}s  observed=${COOL}m  final_replicas=$FINAL" >> "$RUN"
  else
    echo "returned_to_1=no  observed=${COOL}m  final_replicas=$FINAL" >> "$RUN"
  fi
  log "observation done (returned_to_1=$([ -n "$first_at" ] && echo yes || echo no) final=$FINAL)"
fi
echo "COOLDOWN_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"

# 6. Collect + merge + copy (locust CSV + in-cluster autoscaler decision log).
# Re-establish the Prometheus port-forward: a multi-hour observation window can
# drop the long-lived forward, which would fail the range query below.
log "collect: ensuring Prometheus port-forward is live"
# Reuse the existing forward if still serving; only restart if a long observe
# dropped it (restarting needlessly races the port bind).
if ! curl -sf "${PROM}/-/ready" >/dev/null 2>&1; then
  kill "$PF_PID" 2>/dev/null
  for i in $(seq 1 10); do lsof -ti "tcp:${PROM_PORT}" >/dev/null 2>&1 || break; sleep 1; done
  kubectl -n "$NS" port-forward deployment/prometheus "${PROM_PORT}:9090" >>/tmp/${LABEL}_pf.log 2>&1 &
  PF_PID=$!
  for i in $(seq 1 15); do curl -sf "${PROM}/-/ready" >/dev/null 2>&1 && break; sleep 1; done
fi
S=$(grep START_UTC "$RUN"|cut -d= -f2); E=$(grep COOLDOWN_END_UTC "$RUN"|cut -d= -f2)
kubectl -n "$NS" cp "$LG:/tmp/${LABEL}_stats_history.csv" "data/${LABEL}_stats_history.csv" 2>/dev/null \
  && log "collect: locust stats copied" || log "WARN: locust stats copy failed"
if [ "$MODE" = "pred" ]; then
  AP=$(kubectl -n "$NS" get pod -l app="$ASC" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -n "$AP" ] && kubectl -n "$NS" cp "$AP:/tmp/predictive_autoscaler.csv" "logs/${LABEL}_autoscaler.csv" 2>/dev/null
fi
PYTHONPATH=ml "$PY" ml/collect_metrics.py --prometheus-url "$PROM" --start "$S" --end "$E" --output "data/${LABEL}.csv" >>/tmp/${LABEL}_pipe.log 2>&1 \
  && log "collect: prometheus metrics -> data/${LABEL}.csv" || log "ERROR: collect_metrics failed (see /tmp/${LABEL}_pipe.log)"
PYTHONPATH=ml "$PY" ml/merge_locust_prometheus.py --locust-csv "data/${LABEL}_stats_history.csv" --prom-csv "data/${LABEL}.csv" --output "data/${LABEL}_merged.csv" --warmup-minutes 0 >>/tmp/${LABEL}_pipe.log 2>&1 \
  && log "merge: -> data/${LABEL}_merged.csv" || log "ERROR: merge failed (see /tmp/${LABEL}_pipe.log)"
echo "DONE" >> "$RUN"
log "RUN $LABEL COMPLETE"
