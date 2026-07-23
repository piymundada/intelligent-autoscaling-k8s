#!/usr/bin/env bash
# Idle-cost guard. Shows what's running on the experiment clusters and the current
# AWS burn rate, so you don't leave clusters/nodes billing between runs.
#
# Usage:  ./scripts/aws/status.sh
# Env overrides: REGION, HPA_CLUSTER, PRED_CLUSTER, CP_PRICE, NODE_PRICE.
set -u

REGION="${REGION:-us-east-1}"
CLUSTERS=("${HPA_CLUSTER:-thesis-hpa}" "${PRED_CLUSTER:-thesis-pred}")
KCFG_DIR="${KUBECONFIG_DIR:-$HOME/.kube}"
CP_PRICE="${CP_PRICE:-0.10}"      # EKS control plane $/hr per cluster
NODE_PRICE="${NODE_PRICE:-0.105}" # blended 2-vCPU node $/hr incl. Auto Mode fee

aws sts get-caller-identity >/dev/null 2>&1 \
  || { echo "AWS credentials not working — run 'aws configure' or re-paste the lab creds."; exit 1; }

# Host-side hint: a run script mid-flight leaves /tmp/run_*.txt without a DONE line.
inprogress=$(grep -rL DONE /tmp/run_*.txt 2>/dev/null | wc -l | tr -d ' ')

total=0; any_up=0; active=0
printf '%-14s %-9s %-6s %-9s %-8s\n' CLUSTER STATUS NODES APP-PODS '$/hr'
printf '%s\n' "-------------------------------------------------------"

for C in "${CLUSTERS[@]}"; do
  st=$(aws eks describe-cluster --name "$C" --region "$REGION" \
        --query 'cluster.status' --output text 2>/dev/null || echo ABSENT)
  if [ "$st" != "ACTIVE" ]; then
    printf '%-14s %-9s %-6s %-9s %-8s\n' "$C" "${st:-ABSENT}" - - 0.00
    continue
  fi
  any_up=1
  export KUBECONFIG="$KCFG_DIR/$C.config"
  aws eks update-kubeconfig --name "$C" --region "$REGION" >/dev/null 2>&1
  nodes=$(kubectl get nodes --no-headers 2>/dev/null | wc -l | tr -d ' '); nodes=${nodes:-0}
  apps=$(kubectl -n thesis get deploy spring-boot-app -o jsonpath='{.status.readyReplicas}' 2>/dev/null); apps=${apps:-0}
  [ "$apps" -gt 1 ] 2>/dev/null && active=1
  burn=$(awk -v c="$CP_PRICE" -v n="$nodes" -v p="$NODE_PRICE" 'BEGIN{printf "%.2f", c + n*p}')
  total=$(awk -v t="$total" -v b="$burn" 'BEGIN{printf "%.2f", t+b}')
  printf '%-14s %-9s %-6s %-9s %-8s\n' "$C" "$st" "$nodes" "$apps" "$burn"
done
[ "$inprogress" -gt 0 ] 2>/dev/null && active=1

printf '%s\n' "-------------------------------------------------------"
printf 'TOTAL burn: $%s/hr  (~$%s/day if left up)\n' \
  "$total" "$(awk -v t="$total" 'BEGIN{printf "%.2f", t*24}')"

if [ "$any_up" = 0 ]; then
  echo "✓ No clusters up — \$0 burn."
elif [ "$active" = 1 ]; then
  echo "→ A run appears ACTIVE (load in flight and/or app scaled up). OK to keep up."
else
  echo "⚠️  Clusters are UP but no active run detected (app at baseline, no run in flight)."
  echo "   If you're between runs, tear down to stop billing:"
  echo "     for C in ${CLUSTERS[*]}; do CLUSTER_NAME=\$C ./scripts/aws/teardown.sh; done"
fi
