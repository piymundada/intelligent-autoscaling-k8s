#!/usr/bin/env bash
# Run the SAME scenario on BOTH clusters at once, HPA on one, predictive on the
# other: to halve wall-clock vs running the arms sequentially.
#
# Prereqs (once per session): both clusters exist and have been deployed, e.g.
#   CLUSTER_NAME=thesis-hpa  ./scripts/aws/build_push.sh
#   CLUSTER_NAME=thesis-hpa  ./scripts/aws/deploy.sh hpa
#   CLUSTER_NAME=thesis-pred ./scripts/aws/deploy.sh pred
#
# Usage: run_parallel.sh <scenario> <observe_min> <label_prefix> [load_min]
#   observe_min = post-load observation window (NOT a reset). Use 180 (3h) to show
#   HPA staying pinned vs predictive scaling back to the 1-replica baseline.
#   e.g. run_parallel.sh spike 180 spike_run1 60
#        -> data/spike_run1_hpa_*  (cluster thesis-hpa,  prom port 9090)
#           data/spike_run1_pred_* (cluster thesis-pred, prom port 9091)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCEN=$1; COOL=$2; PREFIX=$3; LOAD_MIN=${4:-30}
HPA_CLUSTER="${HPA_CLUSTER:-thesis-hpa}"
PRED_CLUSTER="${PRED_CLUSTER:-thesis-pred}"

echo "==> Parallel run: $SCEN  (hpa=$HPA_CLUSTER, pred=$PRED_CLUSTER, load=${LOAD_MIN}m, observe=${COOL}m)"

# Each arm runs in its own subshell with its own CLUSTER_NAME (=> own kubeconfig
# via env.sh) and a distinct Prometheus port, so they never collide.
CLUSTER_NAME="$HPA_CLUSTER"  "$SCRIPT_DIR/run_eks.sh" "$SCEN" hpa  "$COOL" "${PREFIX}_hpa"  "$LOAD_MIN" 9090 &
PID_HPA=$!
CLUSTER_NAME="$PRED_CLUSTER" "$SCRIPT_DIR/run_eks.sh" "$SCEN" pred "$COOL" "${PREFIX}_pred" "$LOAD_MIN" 9091 &
PID_PRED=$!

echo "    hpa  pid=$PID_HPA  (tail -f /tmp/run_${PREFIX}_hpa.txt)"
echo "    pred pid=$PID_PRED (tail -f /tmp/run_${PREFIX}_pred.txt)"

FAIL=0
wait "$PID_HPA"  || { echo "!! hpa arm exited non-zero";  FAIL=1; }
wait "$PID_PRED" || { echo "!! pred arm exited non-zero"; FAIL=1; }

status(){ grep -q DONE "$1" 2>/dev/null && echo complete || echo INCOMPLETE; }
echo "==> Both arms finished."
echo "    HPA:  $(status /tmp/run_${PREFIX}_hpa.txt)   data/${PREFIX}_hpa_merged.csv"
echo "    PRED: $(status /tmp/run_${PREFIX}_pred.txt)  data/${PREFIX}_pred_merged.csv"
exit $FAIL
