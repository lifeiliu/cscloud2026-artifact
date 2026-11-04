# Pinned manifest-list digest for python:3.11.9-slim-bookworm, resolved 2026-08-30.
FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# E11 uses only the protocol implementation and the pinned cryptography version
# captured in the formal run; it intentionally avoids optional model clients.
RUN pip install --no-cache-dir cryptography==48.0.1
RUN useradd --create-home --shell /bin/bash runner
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner experiments/e10_multiprocess_cross_deployer/ ./experiments/e10_multiprocess_cross_deployer/
COPY --chown=runner:runner experiments/e11_container_cross_deployer/ ./experiments/e11_container_cross_deployer/
USER runner
