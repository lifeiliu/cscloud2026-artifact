FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace
RUN pip install --no-cache-dir cryptography==48.0.1
COPY src/ ./src/
COPY experiments/e10_multiprocess_cross_deployer/ ./experiments/e10_multiprocess_cross_deployer/
COPY experiments/e11_container_cross_deployer/ ./experiments/e11_container_cross_deployer/
RUN useradd --create-home --shell /bin/bash runner
RUN chown -R runner:runner /workspace
USER runner
