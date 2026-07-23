#!/usr/bin/env bash
# Deploy the experiment stack to an existing EKS (Auto Mode) cluster, with the app
# image pointed at ECR. Idempotent, safe to re-run.
#
# Usage:  ./scripts/aws/deploy.sh [base|hpa|pred]   (default: base)
#   base : namespace + nodepool + app + service + observability + loadgen
#          (no autoscaler: bring up the baseline, then add one)
#   hpa  : base + the Kubernetes HPA baseline (k8s/eks/hpa.yaml)
#   pred : base + the in-cluster predictive autoscaler (k8s/eks/autoscaler.yaml)
#
# hpa and pred are mutually exclusive; selecting one removes the other so the two
# autoscalers never fight over the replica count.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODE="${1:-base}"
case "$MODE" in base|hpa|pred) ;; *) echo "usage: deploy.sh [base|hpa|pred]" >&2; exit 1 ;; esac

echo "==> Pointing kubectl at the cluster"
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"

echo "==> Namespace + Auto Mode NodePool"
kubectl apply -f "$ROOT/k8s/namespace.yaml"
# NodePool is a Karpenter (Auto Mode) CRD: skip cleanly on a non-Auto-Mode cluster.
kubectl apply -f "$ROOT/k8s/eks/nodepool.yaml" \
  || echo "  (skipped nodepool — not an Auto Mode cluster?)"

echo "==> App (ECR image), service, observability, loadgen"
kubectl apply -f "$ROOT/k8s/deployment.yaml"
kubectl -n "$NAMESPACE" set image deployment/spring-boot-app "spring-boot-app=$APP_IMAGE"
kubectl -n "$NAMESPACE" patch deployment spring-boot-app --type strategic \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"spring-boot-app","imagePullPolicy":"Always"}]}}}}'
kubectl apply -f "$ROOT/k8s/service.yaml"
kubectl apply -f "$ROOT/k8s/observability/prometheus-configmap.yaml"
kubectl apply -f "$ROOT/k8s/observability/prometheus-deployment.yaml"
kubectl apply -f "$ROOT/k8s/observability/grafana-deployment.yaml"

# In-cluster load generator (Locust scripts mounted from a ConfigMap).
kubectl -n "$NAMESPACE" create configmap locust-scripts \
  --from-file="$ROOT/load-testing/locust/" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/loadgen.yaml"

echo "==> Autoscaler selection: $MODE"
case "$MODE" in
  base)
    kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/eks/hpa.yaml" --ignore-not-found
    kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/eks/autoscaler.yaml" --ignore-not-found
    ;;
  hpa)
    kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/eks/autoscaler.yaml" --ignore-not-found
    kubectl apply -f "$ROOT/k8s/eks/hpa.yaml"
    ;;
  pred)
    kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/eks/hpa.yaml" --ignore-not-found
    kubectl apply -f "$ROOT/k8s/eks/autoscaler.yaml"
    kubectl -n "$NAMESPACE" set image deployment/predictive-autoscaler "autoscaler=$AUTOSCALER_IMAGE"
    # EKS-calibrated knobs (separate from config.py's local values).
    kubectl -n "$NAMESPACE" set env deployment/predictive-autoscaler \
      "RPS_PER_REPLICA=$RPS_PER_REPLICA" "WARMUP_GRACE_CYCLES=$WARMUP_GRACE_CYCLES" "JVM_FLOOR=$JVM_FLOOR"
    ;;
esac

echo "==> Waiting for app rollout"
kubectl -n "$NAMESPACE" rollout status deployment/spring-boot-app --timeout=180s

echo "==> Done."
echo "    Prometheus:  kubectl -n $NAMESPACE port-forward deployment/prometheus 9090:9090"
[ "$MODE" = "pred" ] && echo "    Autoscaler logs:  kubectl -n $NAMESPACE logs -f deployment/predictive-autoscaler"
