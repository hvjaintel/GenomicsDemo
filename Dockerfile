# Booth UI container (optional — ./run_demo.sh is the simpler path).
#
# This image contains ONLY the Gradio front end. DeepVariant runs as a sibling
# container on the host via the mounted Docker socket, so it keeps direct
# access to the CPU's AMX units and the real NUMA topology.

FROM python:3.10-slim

# The Docker CLI is the app's interface to the host daemon.
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io numactl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .

EXPOSE 7860

CMD ["python", "-m", "app.main"]
