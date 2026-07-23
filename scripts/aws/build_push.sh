#!/usr/bin/env bash
# Build the app + predictive-autoscaler images (linux/amd64 for EKS x86 nodes) and
# push them to ECR. Creates the ECR repositories if they don't exist.
#
# Usage:  ./scripts/aws/build_push.sh [app|autoscaler|all]   (default: all)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TARGET="${1:-all}"

ensure_repo() {
  aws ecr describe-repositories --repository-names "$1" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$1" --region "$REGION" >/dev/null
}

echo "==> ECR login"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

if [ "$TARGET" = "app" ] || [ "$TARGET" = "all" ]; then
  echo "==> Building app image (Maven package + docker build)"
  ensure_repo "$APP_REPO_NAME"
  ( cd "$ROOT/spring-workload-simulator" \
      && mvn -q -DskipTests package \
      && docker build --platform linux/amd64 -t "$APP_IMAGE" . )
  docker push "$APP_IMAGE"
fi

if [ "$TARGET" = "autoscaler" ] || [ "$TARGET" = "all" ]; then
  echo "==> Building predictive-autoscaler image"
  ensure_repo "$AUTOSCALER_REPO_NAME"
  if [ ! -f "$ROOT/ml/models/lstm_forecaster.pt" ] || [ ! -f "$ROOT/data/processed/scaler.json" ]; then
    echo "ERROR: trained model/scaler missing — run ml/train.py first." >&2
    echo "       (expected ml/models/lstm_forecaster.pt and data/processed/scaler.json)" >&2
    exit 1
  fi
  docker build --platform linux/amd64 -f "$ROOT/ml/Dockerfile" -t "$AUTOSCALER_IMAGE" "$ROOT"
  docker push "$AUTOSCALER_IMAGE"
fi

echo "==> Done. Pushed:"
[ "$TARGET" != "autoscaler" ] && echo "    $APP_IMAGE"
[ "$TARGET" != "app" ]        && echo "    $AUTOSCALER_IMAGE"
