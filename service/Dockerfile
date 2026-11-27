# The vision service (design note 40, phase 3): fast photo checks, CPU only, no internet at run time.
# Build context: this folder (vision/). Model files are NOT in the image: they live in the
# vision_models volume, filled once by deploy/fetch-models.sh and checked against models.lock.json.
# Dependencies are locked with their sha256 (requirements.lock); pdqhash has no Linux wheel and is
# compiled from its locked sources in the first stage, on amd64 and arm64 alike.

# python:3.12.16-slim-trixie, multi-platform index (linux/amd64 and linux/arm64), pinned by digest.
FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*
COPY requirements-build.lock requirements.lock /tmp/
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/requirements-build.lock \
 && /opt/venv/bin/pip install --no-cache-dir --require-hashes --no-build-isolation --no-binary pdqhash -r /tmp/requirements.lock \
 && /opt/venv/bin/pip uninstall -y cython setuptools wheel \
 && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} +

FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
COPY --from=build /opt/venv /opt/venv
# MALLOC_ARENA_MAX: fewer glibc arenas, less memory kept by threads. One thread for OpenMP-style pools.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MODELS_DIR=/models
# A named volume takes the owner of the folder it is mounted on: /models must exist, owned by vision
# (same uid as the API's user), so deploy/fetch-models.sh can write into it.
RUN useradd --system --uid 1001 --no-create-home vision \
 && mkdir -p /models \
 && chown vision /models
WORKDIR /app
COPY models.lock.json ./
COPY app ./app
USER vision
EXPOSE 8080
CMD ["python", "-m", "app.server"]
