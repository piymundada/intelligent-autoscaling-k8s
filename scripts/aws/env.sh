#!/usr/bin/env bash
# Shared AWS/EKS configuration, sourced by build_push.sh / deploy.sh / teardown.sh.
#
# Override any value via the environment, e.g.:
#   REGION=eu-west-1 CLUSTER_NAME=my-thesis ./scripts/aws/deploy.sh
#
# Requires the AWS CLI to be configured (aws sts get-caller-identity must work).
set -euo pipefail

REGION="${REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-thesis-autoscaling}"
NAMESPACE="${NAMESPACE:-thesis}"
APP_REPO_NAME="${APP_REPO_NAME:-spring-workload-simulator}"
AUTOSCALER_REPO_NAME="${AUTOSCALER_REPO_NAME:-predictive-autoscaler}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# EKS-calibrated sustainable RPS per pod (from the EKS saturation ramp +
# calibrate_capacity.py). SEPARATE from config.py's local/Minikube value, set it
# here once calibrated on EKS; deploy.sh injects it into the predictive autoscaler.
RPS_PER_REPLICA="${RPS_PER_REPLICA:-18}"

# EKS warm-up grace: suppress the JVM-CPU floor for the first N cycles while the
# JVM JIT-compiles (EKS container ~1 core -> warmup CPU 60-95% falsely trips it).
# 0 locally; ~3 on EKS. SEPARATE from config.py. deploy.sh injects it.
WARMUP_GRACE_CYCLES="${WARMUP_GRACE_CYCLES:-3}"

# JVM-pressure floor: "on" = JVM-aware arm, "off" = RPS-only ablation (RQ2).
JVM_FLOOR="${JVM_FLOOR:-on}"

# Repo root = two levels up from this file (scripts/aws/env.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Per-cluster kubeconfig so two clusters (e.g. thesis-hpa + thesis-pred) can be
# driven in parallel from one machine without racing a shared current-context.
# `aws eks update-kubeconfig` and all kubectl in this shell honour this env var.
export KUBECONFIG="${KUBECONFIG_DIR:-$HOME/.kube}/${CLUSTER_NAME}.config"
mkdir -p "$(dirname "$KUBECONFIG")"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
APP_IMAGE="${REGISTRY}/${APP_REPO_NAME}:${IMAGE_TAG}"
AUTOSCALER_IMAGE="${REGISTRY}/${AUTOSCALER_REPO_NAME}:${IMAGE_TAG}"

echo "  region=${REGION}  account=${ACCOUNT}  cluster=${CLUSTER_NAME}"
echo "  app_image=${APP_IMAGE}"
echo "  autoscaler_image=${AUTOSCALER_IMAGE}"
