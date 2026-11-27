"""Send photos to the vlm service with the strict JSON schema of plan phase 5, check every reply.

Runs in a throw-away container of the vision image (it has Pillow) that shares the vlm
container's network, so vlm needs no published port:

    docker run --rm --network container:<vlm container> --user "$(id -u)" \
      -v "$PWD/service:/w:ro" -v "$HOME/.cache/dlala-vision-tests:/cache" -e VISION_TEST_CACHE=/cache \
      -w /w <vision image> python -m vlm.test_schema [--images 20] [--side 640]

Exit code 0 only when every reply parses as JSON and fits the schema. Prints the time of each
answer and llama-server's own timings (prompt and generation).
"""
import argparse
import base64
import http.client
import io
import json
import sys
import time

from PIL import Image

sys.path.insert(0, "/w")
from tests.fixtures import fixture_path, load_fixtures  # noqa: E402

PROMPT_VERSION = "photo-vlm-v1"
# The overview sent to the model: longest side in px (phase 1 picks the production value).
DEFAULT_SIDE = 640
# Short answers: three fields, the evidence at most EVIDENCE_MAX characters.
MAX_TOKENS = 120
EVIDENCE_MAX = 160
BOX_SCALE = 1000  # box coordinates are integers from 0 to 1000, relative to the image

# Rules (phase 0, point 2; llama.cpp common/json-schema.cpp at b12679): integer bounds only on
# integers, no anyOf beside properties in the same object, the format repeated in the prompt.
SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "box": {"anyOf": [
            {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": BOX_SCALE}, "minItems": 4, "maxItems": 4},
            {"type": "null"},
        ]},
        "evidence": {"type": "string", "maxLength": EVIDENCE_MAX},
    },
    "required": ["present", "box", "evidence"],
    "additionalProperties": False,
}
# SAMPLE instruction and question: they show the call's format, not the production wording. The production
# instructions and the per-group questions (with their expert cues) are private: they live in the API, are
# versioned (PROMPT_VERSION) and are not published, so they cannot be studied to word a photo around them.
SYSTEM = "You answer one closed question about a photo, only with JSON."
QUESTION = (
    "Does this photo show a chair?\n"
    "Reply with JSON only, exactly these keys: "
    '{"present": "yes" or "no" or "unsure", '
    f'"box": [x0, y0, x1, y1] around the chair as integers from 0 to {BOX_SCALE} relative to the image width and height, or null, '
    f'"evidence": a short reason, at most 20 words}}'
)


def check(answer):
    """The reply against SCHEMA (by hand: the image has no jsonschema package). Raises ValueError."""
    if not isinstance(answer, dict) or set(answer) != {"present", "box", "evidence"}:
        raise ValueError(f"keys {sorted(answer) if isinstance(answer, dict) else type(answer)}")
    if answer["present"] not in ("yes", "no", "unsure"):
        raise ValueError(f"present={answer['present']!r}")
    box = answer["box"]
    if box is not None:
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= BOX_SCALE for v in box)):
            raise ValueError(f"box={box!r}")
    if not isinstance(answer["evidence"], str) or len(answer["evidence"]) > EVIDENCE_MAX:
        raise ValueError("evidence")


def ask(host, port, jpeg_bytes):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                # The image before the text (phase 0, point 2).
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}},
                {"type": "text", "text": QUESTION},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "photo_vlm_v1", "strict": True, "schema": SCHEMA}},
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    conn = http.client.HTTPConnection(host, port, timeout=900)
    t = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    ms = (time.perf_counter() - t) * 1000
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {raw[:300]!r}")
    return json.loads(raw), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--side", type=int, default=DEFAULT_SIDE)
    a = ap.parse_args()
    fixtures = [f for f in load_fixtures() if f["kind"] in ("photo", "face")][: a.images]
    failures = 0
    for f in fixtures:
        im = Image.open(fixture_path(f)).convert("RGB")
        im.thumbnail((a.side, a.side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        try:
            resp, ms = ask(a.host, a.port, buf.getvalue())
            text = resp["choices"][0]["message"]["content"]
            answer = json.loads(text)
            check(answer)
            t = resp.get("timings", {})
            print(f"ok    {f['id']:<16} {ms:7.0f} ms  prompt {t.get('prompt_n')} tok {t.get('prompt_ms', 0):6.0f} ms  gen {t.get('predicted_n')} tok {t.get('predicted_ms', 0):6.0f} ms  {json.dumps(answer)}", flush=True)
        except (ValueError, KeyError, RuntimeError, json.JSONDecodeError) as e:
            failures += 1
            print(f"FAIL  {f['id']:<16} {type(e).__name__}: {e}", flush=True)
    print(f"{len(fixtures) - failures} of {len(fixtures)} replies parse and fit the schema ({PROMPT_VERSION})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
