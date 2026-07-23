#!/usr/bin/env bash
# One experiment run: fresh pods -> load (30m) -> cooldown -> collect/merge/copy.
#
# Usage: run_local.sh <scenario> <mode> <cooldown_min> <label>
#   scenario : steady | bursty | spike | gc
#   mode     : hpa | pred | pred-rpsonly
#              (pred* starts ml/autoscaler.py, needs trained model;
#               pred-rpsonly adds --no-jvm-floor = the ablation arm)
#   cooldown : 0 = none; else minutes (monitor until 1 replica OR this cap)
#   label    : output prefix, e.g. hpa_spike_run1
#
# Writes:  data/<label>{,_stats_history,_merged}.csv  and  /tmp/run_<label>.txt
set -u
cd /Users/ppmundada/IdeaProjects/intelligent-autoscaling-k8s
SCEN=$1; MODE=$2; COOL=$3; LABEL=$4; LOAD_MIN=${5:-30}
NS=thesis
RUN=/tmp/run_${LABEL}.txt; : > "$RUN"
LG=$(kubectl -n $NS get pod -l app=loadgen -o jsonpath='{.items[0].metadata.name}')

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN"; }

log "RUN $LABEL ($SCEN/$MODE cooldown=${COOL}m)"

# 1. GUARANTEE a fresh, single-replica start every run.
#    Remove HPA, scale to ZERO (terminate ALL old/high-memory pods so none linger
#    and trick HPA into scaling up), then scale to 1 brand-new pod (fresh JVM ->
#    low memory, CPU idle). HPA is applied only AFTER the clean 1-pod state.
kubectl -n $NS delete hpa spring-boot-hpa --ignore-not-found >/dev/null 2>&1
kubectl -n $NS scale deploy/spring-boot-app --replicas=0 >/dev/null 2>&1
t=0; until [ "$(kubectl -n $NS get pods -l app=spring-boot-app --no-headers 2>/dev/null | wc -l | tr -d ' ')" = "0" ] || [ $t -ge 36 ]; do sleep 5; t=$((t+1)); done
kubectl -n $NS scale deploy/spring-boot-app --replicas=1 >/dev/null 2>&1
kubectl -n $NS rollout status deploy/spring-boot-app --timeout=180s >/dev/null 2>&1
t=0; until [ "$(kubectl -n $NS get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')" = "1" ] || [ $t -ge 36 ]; do sleep 5; t=$((t+1)); done
sleep 12  # let metrics-server report the fresh pod
MEM=$(kubectl -n $NS top pod -l app=spring-boot-app --no-headers 2>/dev/null | awk '{print $3}' | head -1)
NP=$(kubectl -n $NS get pods -l app=spring-boot-app --no-headers 2>/dev/null | wc -l | tr -d ' ')
log "fresh start: pods=$NP mem=$MEM (should be 1 pod, ~250-300Mi)"

# 2. enable the autoscaler under test (HPA mode only; pred starts its loop in step 4)
if [ "$MODE" = "hpa" ]; then
  kubectl apply -f k8s/hpa.yaml >/dev/null 2>&1
fi

# 4. predictive: launch the control loop on the host (reads prom :9090, kubectl scale)
APID=""
if [ "$MODE" = "pred" ] || [ "$MODE" = "pred-rpsonly" ]; then
  FLOOR_FLAG=""
  [ "$MODE" = "pred-rpsonly" ] && FLOOR_FLAG="--no-jvm-floor"
  PYTHONPATH=ml nohup /usr/bin/python3 ml/autoscaler.py \
    --prometheus-url http://localhost:9090 \
    --model ml/models/lstm_forecaster.pt \
    --scaler data/processed/scaler.json \
    --rps-per-replica 35 --max-replicas 4 --control-interval 30 \
    --scale-down-delay-cycles 16 $FLOOR_FLAG \
    --log-file logs/${LABEL}_autoscaler.csv >/tmp/${LABEL}_autoscaler.log 2>&1 &
  APID=$!; sleep 5
fi

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"
echo "start_replicas=$(kubectl -n $NS get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')" >> "$RUN"
log "load start"

# 5. in-cluster load (30 min)
# Re-fetch the loadgen pod name now (it may have been restarted since script start,
# which would make the cached name stale -> NotFound -> load silently skipped).
kubectl -n $NS rollout status deploy/loadgen --timeout=120s >/dev/null 2>&1
LG=$(kubectl -n $NS get pod -l app=loadgen --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
# steady has no LoadTestShape -> MUST pass --users (else locust defaults to 1).
# bursty/spike have LoadTestShape -> it controls users; --users is ignored but
# passed as a harmless nominal.
case "$SCEN" in
  steady) USERS="--users 20 --spawn-rate 2" ;;
  bursty) USERS="--users 180 --spawn-rate 40" ;;
  spike)  USERS="--users 180 --spawn-rate 60" ;;
  gc)     USERS="--users 80 --spawn-rate 20" ;;
  *)      USERS="--users 20 --spawn-rate 2" ;;
esac
kubectl -n $NS exec "$LG" -- sh -c "rm -f /tmp/${LABEL}*" 2>/dev/null
# LOAD_SECONDS drives the LoadTestShape stop time (locust ignores --run-time
# when a shape is active); --run-time still covers the shapeless steady scenario.
kubectl -n $NS exec "$LG" -- sh -c "LOAD_SECONDS=$((LOAD_MIN*60)) locust -f /mnt/locust/locustfile_${SCEN}.py \
  --host http://spring-boot-app --headless --run-time ${LOAD_MIN}m $USERS \
  --csv /tmp/${LABEL} --csv-full-history" >/tmp/${LABEL}_locust.log 2>&1
echo "LOAD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"
log "load done"

# 6. cooldown: until 1 replica OR cap
if [ "$COOL" -gt 0 ]; then
  c=0; CAP=$((COOL*4))
  until [ "$(kubectl -n $NS get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')" = "1" ] || [ $c -ge $CAP ]; do sleep 15; c=$((c+1)); done
  RET=$([ "$(kubectl -n $NS get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}')" = "1" ] && echo yes || echo no)
  echo "returned_to_1=$RET  after=$((c*15))s" >> "$RUN"
  log "cooldown done (returned_to_1=$RET)"
fi
echo "COOLDOWN_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN"

# 7. stop predictive autoscaler
[ -n "$APID" ] && kill "$APID" 2>/dev/null

# 8. collect + merge + copy
S=$(grep START_UTC "$RUN"|cut -d= -f2); E=$(grep COOLDOWN_END_UTC "$RUN"|cut -d= -f2)
kubectl -n $NS cp "$LG:/tmp/${LABEL}_stats_history.csv" "data/${LABEL}_stats_history.csv" 2>/dev/null
PYTHONPATH=ml /usr/bin/python3 ml/collect_metrics.py --prometheus-url http://localhost:9090 --start "$S" --end "$E" --output "data/${LABEL}.csv" >>/tmp/${LABEL}_pipe.log 2>&1
PYTHONPATH=ml /usr/bin/python3 ml/merge_locust_prometheus.py --locust-csv "data/${LABEL}_stats_history.csv" --prom-csv "data/${LABEL}.csv" --output "data/${LABEL}_merged.csv" --warmup-minutes 0 >>/tmp/${LABEL}_pipe.log 2>&1
echo "DONE" >> "$RUN"
log "RUN $LABEL COMPLETE"
