#!/usr/bin/env bash
# The vision tests, in one command, inside the vision image (built here for this computer's platform).
#
#   MODELS_DIR=<folder filled by app.fetch_models> service/tests/run.sh [unittest arguments]
#
# Fill the models folder once (on a work computer, with internet):
#   python3 -m app.fetch_models --lock models.lock.json --dest <folder> --groups vision vision-dino-base   (from service/)
# Test images are downloaded once into VISION_TEST_CACHE (default ~/.cache/dlala-vision-tests) and
# checked by sha256 (tests/fixtures.json). Takes a few minutes: the models run on one thread.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
models=${MODELS_DIR:?set MODELS_DIR to the folder that holds vision/... model files}
cache=${VISION_TEST_CACHE:-$HOME/.cache/dlala-vision-tests}
image=${VISION_TEST_IMAGE:-dlala-vision:test}
mkdir -p "$cache"
docker build -q -t "$image" "$here" > /dev/null
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$models:/models:ro" -v "$here/tests:/app/tests:ro" -v "$cache:/cache" \
  -e VISION_TEST_CACHE=/cache "$image" \
  python -m unittest discover -s tests -t . "$@"
