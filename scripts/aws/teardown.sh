#!/usr/bin/env bash
# Remove the experiment workloads to stop EC2/node billing between sessions.
# By default it deletes the whole `thesis` namespace (everything in it) plus the
# custom NodePool, so Auto Mode consolidates the experiment nodes away.
#
# This does NOT delete the EKS cluster itself (control plane = $0.10/hr). To stop
# ALL billing, delete the cluster afterwards, see the printed reminder.
#
# Usage:  ./scripts/aws/teardown.sh [--keep-namespace]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION" >/dev/null

if [ "${1:-}" = "--keep-namespace" ]; then
  echo "==> Deleting workloads (keeping namespace)"
  kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/eks/autoscaler.yaml" --ignore-not-found
  kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/hpa.yaml" --ignore-not-found
  kubectl -n "$NAMESPACE" delete -f "$ROOT/k8s/loadgen.yaml" --ignore-not-found
  kubectl -n "$NAMESPACE" delete deployment spring-boot-app prometheus grafana --ignore-not-found
else
  echo "==> Deleting the $NAMESPACE namespace (all workloads)"
  kubectl delete namespace "$NAMESPACE" --ignore-not-found
fi

echo "==> Deleting the experiment NodePool (Auto Mode consolidates its nodes)"
kubectl delete -f "$ROOT/k8s/eks/nodepool.yaml" --ignore-not-found

cat <<EOF

==> Workloads removed. To stop ALL billing, delete the cluster too:
      eksctl delete cluster --name $CLUSTER_NAME --region $REGION
    (or via the EKS console). Auto Mode re-provisions nodes in ~1 min next time.
EOF
