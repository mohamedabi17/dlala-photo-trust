"""Memory and time of a running vision service (plan phase 3: p50/p95 on the ARM server).

Runs in a throw-away container of the vision image that shares the service's network, so the
service needs no published port (on the server: the container name of `vision`):

    docker run --rm --network container:<vision container> --user "$(id -u)" \
      -v "$PWD/service/tests:/app/tests:ro" -v "$HOME/.cache/dlala-vision-tests:/cache" -e VISION_TEST_CACHE=/cache \
      <vision image> python -m tests.measure [--runs 5]

Prints the process memory (VmRSS, and VmHWM = peak) after each step, then the time of each route.
Start it on a freshly started service: the first steps measure what each model adds.
"""
import argparse
import http.client
import io
import json
import statistics
import time

from PIL import Image

from .common import frames, jpeg, load, pil

# The largest original phase 2 keeps (4096 px a side), and a typical phone photo.
LARGE_SIDE = 4096
TYPICAL_SIDE = 1600


def call(url, method, path, body=b""):
    host, port = url.split("//")[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=900)
    t = time.perf_counter()
    conn.request(method, path, body, {"Content-Type": "application/x-dlala-frames"} if body else {})
    r = conn.getresponse()
    data = r.read()
    ms = (time.perf_counter() - t) * 1000
    conn.close()
    if r.status != 200:
        raise SystemExit(f"{method} {path}: {r.status} {data[:300]!r}")
    return (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data), ms


def resized(data, side):
    im = pil(data)
    s = side / max(im.size)
    return jpeg(im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS), 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    face = load("wm-sally-ride")
    large, typical = resized(face, LARGE_SIDE), resized(face, TYPICAL_SIDE)
    other = resized(load("wm-33284613"), TYPICAL_SIDE)

    def mem(step):
        h, _ = call(a.url, "GET", "/health")
        print(f"  {step:<44} rss {h['rss_mb']:>7} MB   peak {h['peak_rss_mb']:>7} MB   models {h['models']}")

    print("memory (process of the service):")
    mem("idle")
    call(a.url, "POST", "/fingerprint", frames({}, typical))
    mem("after /fingerprint (DINOv2)")
    call(a.url, "POST", "/screen", frames({"parts": ["faces"]}, typical))
    mem("after /screen faces (YuNet)")
    call(a.url, "POST", "/screen", frames({"parts": ["siglip"], "tiles": 0}, typical))
    mem("after /screen siglip (SigLIP 2)")
    call(a.url, "POST", "/screen", frames({"parts": ["ocr"]}, typical))
    mem("after /screen ocr (RapidOCR)")
    call(a.url, "POST", "/screen", frames({"parts": ["siglip", "hotspots", "ocr", "faces", "photo_rects"]}, large))
    mem(f"after /screen, all parts, {LARGE_SIDE} px")
    box = [1000, 700, 1900, 1800]
    cases = [
        ("/fingerprint", f"{TYPICAL_SIDE} px", frames({}, typical)),
        ("/fingerprint", f"{LARGE_SIDE} px + 1 region", frames({"regions": [[v * 4096 // 3487 for v in box]]}, large)),
        ("/screen", f"{TYPICAL_SIDE} px, default", frames({}, typical)),
        ("/screen", f"{LARGE_SIDE} px, default", frames({}, large)),
        ("/screen", f"{TYPICAL_SIDE} px, hotspots only", frames({"parts": ["hotspots"]}, typical)),
        ("/verify", f"{TYPICAL_SIDE} px pair", frames({}, typical, other)),
        ("/redact", f"{LARGE_SIDE} px, 1 box", frames({"boxes": [[v * 4096 // 3487 for v in box]]}, large)),
    ]
    print(f"time per call ({a.runs} runs each):")
    for path, label, body in cases:
        times, inner = [], None
        for _ in range(a.runs):
            out, ms = call(a.url, "POST", path, body)
            times.append(ms)
            if isinstance(out, dict):
                inner = out.get("timings_ms")
        times.sort()
        p95 = times[min(len(times) - 1, round(0.95 * (len(times) - 1)))]
        print(f"  {path:<13} {label:<28} p50 {statistics.median(times):>7.0f} ms   p95 {p95:>7.0f} ms" + (f"   {inner}" if inner else ""))
    mem("at the end")


if __name__ == "__main__":
    main()
