# Predictive autoscaler — in-cluster controller image (the JVM-aware arm).
#
# Runs ml/autoscaler.py as a long-lived Deployment on EKS. It reads metrics from
# the in-cluster Prometheus service and scales spring-boot-app via `kubectl scale`,
# using the pod's ServiceAccount token automatically (kubectl detects in-cluster
# config when KUBERNETES_SERVICE_HOST is set).
#
# Build from the REPO ROOT (the context needs both ml/ and data/processed/):
#   docker build --platform linux/amd64 -f ml/Dockerfile -t <ecr-repo>:latest .
# (scripts/aws/build_push.sh does this for you.)
FROM python:3.11-slim

# kubectl for the get/scale subprocess calls in autoscaler.py.
ARG KUBECTL_VERSION=v1.30.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -sSLo /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA). torch comes from the dedicated
# CPU wheel index; numpy/requests from PyPI.
# numpy pinned <2: torch 2.2.2 is built against the NumPy 1.x ABI and crashes on
# import under NumPy 2.x.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install --no-cache-dir "numpy>=1.26.0,<2" "requests>=2.31.0"

# Code + trained artefacts baked in (models gitignored but present locally).
COPY ml/ /app/ml/
# Use the EKS-trained model + scaler in the container (overwrites the Minikube
# model that came via `COPY ml/` at the same container path). The repo keeps both
# models — ml/models/lstm_forecaster.pt (Minikube) and lstm_forecaster_eks.pt — and
# config.py's local values are untouched. EKS-specific paths, set in the retrain.
COPY ml/models/lstm_forecaster_eks.pt /app/ml/models/lstm_forecaster.pt
COPY data/processed_eks/scaler.json /app/artifacts/scaler.json

ENV PYTHONPATH=/app/ml PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/ml/autoscaler.py"]
