"""Settings of the vision service, read once from the environment (no secrets: there are none)."""
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Version of the HTTP contract (vision/README.md). The API checks it in GET /health.
API_VERSION = "vision-v1"

# Where deploy/fetch-models.sh put the model files (the vision_models volume, read-only here).
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
# The lock that names every model file, its size and sha256 (copied into the image).
LOCK_PATH = os.environ.get("LOCK_PATH", os.path.join(os.path.dirname(APP_DIR), "models.lock.json"))
# Precomputed SigLIP 2 text vectors (tools/build-prompts.py) and the optional logistic probe (phase 1).
PROMPTS_DIR = os.environ.get("PROMPTS_DIR", os.path.join(APP_DIR, "data"))
PROBE_PATH = os.environ.get("PROBE_PATH", os.path.join(APP_DIR, "data", "probe.json"))

# DINOv2 size: "small" (384 values, less memory) or "base" (768 values); phase 1 picks one.
DINO_VARIANT = os.environ.get("DINO_VARIANT", "small")

# Threads per ONNX Runtime session: the container has cpus 0.75, so one is enough.
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))
# Re-hash each model file when it is first loaded (catches a damaged volume).
VERIFY_SHA_ON_LOAD = os.environ.get("VERIFY_SHA_ON_LOAD", "1") == "1"

HOST = os.environ.get("VISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISION_PORT", "8080"))
# Requests computed at once. One: with cpus 0.75 a second request in parallel adds memory, not
# speed; the API's second photo waits its turn (QUEUE_WAIT_S), then gets 503 busy and retries.
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "1"))
# How long a request waits for a free slot before the answer is 503 busy.
QUEUE_WAIT_S = float(os.environ.get("QUEUE_WAIT_S", "60"))
# Requests allowed to wait (each holds its body, up to MAX_BODY_BYTES): beyond, 503 busy at once,
# before the body is read.
MAX_WAITING = int(os.environ.get("MAX_WAITING", "4"))
# After this many seconds without a request, the models leave memory (about 0.6 GB back to the
# server when nobody sells); the next request loads them again. 0 keeps them loaded.
IDLE_UNLOAD_S = float(os.environ.get("VISION_IDLE_UNLOAD_S", "600"))
# Largest request body: two originals of at most 4096 px in JPEG quality 90 fit easily.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(32 * 1024 * 1024)))
# Seconds allowed to receive a request before the connection is dropped.
SOCKET_TIMEOUT_S = float(os.environ.get("SOCKET_TIMEOUT_S", "30"))
