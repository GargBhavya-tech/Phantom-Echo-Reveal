# PHANTOM-ECHO REVEAL — one-command containerised demo
# Build:  docker build -t phantom-echo .
# Run:    docker run -p 8000:8000 phantom-echo            # live dashboard
# Eval:   docker run phantom-echo python -m src.eval.run_real_eval \
#                    --dataset datasets/redwood_sample --frames 4
# Tests:  docker run phantom-echo python -m pytest tests/ -q
#
# Everything runs offline on CPU in the default simulate mode — no GPU, no
# cloud keys, no model downloads. The real-data KPI reads datasets/redwood_sample
# which is committed in the image.

FROM python:3.12-slim

# System libs needed by opencv-python-headless / open3d at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Default to offline simulate mode so the container is self-contained.
ENV PHANTOM_SIMULATE=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install deps first (layer cache) then copy source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sanity-check the build: the full test suite must pass at image-build time.
RUN python -m pytest tests/ -q

EXPOSE 8000

# Default command: the live dashboard at http://localhost:8000
CMD ["python", "-m", "src.main", "--mode", "realtime"]
