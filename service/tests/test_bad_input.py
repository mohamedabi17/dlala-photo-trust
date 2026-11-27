"""Every route on bad input: a clean JSON error with the right status, never a crash."""
import io
import os
import struct
import tempfile
import threading
import time
import unittest
import zlib

from PIL import Image

from app import settings

from .common import ServerCase, frames, jpeg, load

ROUTES = {"/fingerprint": 1, "/screen": 1, "/verify": 2, "/redact": 1}
# Each route's minimal valid meta, so the error tested is the image's.
META = {"/fingerprint": {}, "/screen": {"parts": ["faces"]}, "/verify": {}, "/redact": {"boxes": [[0, 0, 10, 10]]}}


def png_claiming(width, height):
    """A tiny PNG whose header claims width x height (a decompression bomb's first bytes)."""
    buf = io.BytesIO()
    Image.new("L", (1, 1)).save(buf, format="PNG")
    data = bytearray(buf.getvalue())
    data[16:24] = struct.pack(">II", width, height)  # IHDR width and height
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))  # IHDR CRC over type + data
    return bytes(data)


class BadInput(ServerCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.good = jpeg(Image.open(io.BytesIO(load("abo-0007ef0a"))))

    def expect(self, status_code, error_code, path, body, ctype="application/x-dlala-frames", headers=None):
        status, out, _ = self.request("POST", path, body, ctype, headers)
        self.assertEqual(status, status_code, f"{path}: {out}")
        self.assertEqual(out["error"]["code"], error_code, path)

    def images(self, path, bad):
        return [bad] + [self.good] * (ROUTES[path] - 1)

    def test_empty_body(self):
        for path in ROUTES:
            self.expect(400, "empty_body", path, b"")

    def test_empty_image_part(self):
        for path in ROUTES:
            self.expect(400, "empty_image", path, frames(META[path], *self.images(path, b"")))

    def test_not_an_image(self):
        for path in ROUTES:
            self.expect(415, "not_an_image", path, frames(META[path], *self.images(path, b"this is not an image " * 50)))
            if ROUTES[path] == 1:  # a bare body carries one image
                self.expect(415, "not_an_image", path, b"%PDF-1.7 not an image", "image/jpeg")

    def test_truncated_image(self):
        big = load("wm-33284613")
        for path in ROUTES:
            self.expect(415, "not_an_image", path, frames(META[path], *self.images(path, big[: len(big) // 3])))

    def test_huge_image(self):
        bomb = png_claiming(30000, 30000)
        for path in ROUTES:
            self.expect(413, "image_too_large", path, frames(META[path], *self.images(path, bomb)))
        self.expect(413, "image_too_large", "/screen", png_claiming(9000, 100), "image/png")

    def test_long_thin_images(self):
        # 130:1 and more: the finder's 400 px copy is 1 px thick (it once answered 500).
        for size in ((8192, 16), (16, 8192)):
            body = jpeg(Image.new("RGB", size, (200, 120, 40)))
            for path, meta in (("/screen", {}), ("/screen", {"parts": ["photo_rects"]}), ("/fingerprint", {"tiles": True})):
                status, out, _ = self.request("POST", path, frames(meta, body))
                self.assertEqual(status, 200, f"{path} {meta} {size}: {out}")
                if path == "/fingerprint":
                    self.assertEqual(out["regions"], [])  # no tiles below 64 px a side
                else:
                    self.assertEqual(out["photo_rects"], [])

    def test_body_too_large(self):
        for path in ROUTES:
            # Only the header is sent: the server answers without reading the body.
            self.expect(413, "body_too_large", path, b"", headers={"Content-Length": str(settings.MAX_BODY_BYTES + 1)})

    def test_framing_and_meta(self):
        self.expect(415, "bad_content_type", "/screen", self.good, "text/plain")
        self.expect(400, "bad_frames", "/screen", struct.pack(">I", 10_000) + b"{}")
        self.expect(400, "bad_frames", "/screen", b"\x00\x00")
        bad_json = b"{not json"
        self.expect(400, "bad_meta", "/screen", struct.pack(">I", len(bad_json)) + bad_json + self.good)
        self.expect(400, "bad_meta", "/screen", struct.pack(">I", 2) + b"[]" + self.good)
        self.expect(400, "bad_frames", "/verify", frames({}, self.good))
        self.expect(400, "bad_frames", "/screen", frames({}, self.good, self.good))
        self.expect(400, "bad_meta", "/redact", frames({}, self.good))
        self.expect(400, "bad_meta", "/redact", frames({"boxes": [[1, 2, 3]]}, self.good))
        self.expect(400, "bad_meta", "/redact", frames({"boxes": [[50, 50, 10, 10]]}, self.good))
        self.expect(400, "bad_meta", "/redact", frames({"boxes": [["a", 0, 1, 1]]}, self.good))
        self.expect(400, "bad_meta", "/screen", frames({"parts": ["everything"]}, self.good))
        self.expect(400, "bad_meta", "/screen", frames({"tiles": 99}, self.good))
        self.expect(400, "bad_meta", "/fingerprint", frames({"regions": [[0, 0, 5, 5]]}, self.good))
        self.expect(400, "bad_meta", "/verify", frames({"box_a": "x"}, self.good, self.good))
        self.expect(400, "image_too_small", "/screen", jpeg(Image.new("RGB", (8, 8))), "image/jpeg")
        self.expect(411, "length_required", "/screen", self.good, headers={"Transfer-Encoding": "chunked", "Content-Length": ""})

    def test_boxes_not_finite(self):
        # Python's json reads NaN and Infinity: they must be refused, not reach int().
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.expect(400, "bad_meta", "/redact", frames({"boxes": [[bad, 0, 10, 10]]}, self.good))
            self.expect(400, "bad_meta", "/fingerprint", frames({"regions": [[0, 0, 100, bad]]}, self.good))
            self.expect(400, "bad_meta", "/verify", frames({"box_a": [0, bad, 50, 50]}, self.good, self.good))

    def test_formats_not_parsed(self):
        # Only JPEG, PNG, WebP (and MPO) are opened; TIFF, BMP and GIF are refused unparsed.
        for fmt in ("TIFF", "BMP", "GIF"):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, format=fmt)
            for path in ROUTES:
                self.expect(415, "not_an_image", path, frames(META[path], *self.images(path, buf.getvalue())))
        for fmt in ("PNG", "WEBP"):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, format=fmt)
            status, out, _ = self.request("POST", "/redact", frames(META["/redact"], buf.getvalue()))
            self.assertEqual(status, 200, f"{fmt}: {out}")

    def test_unknown_route(self):
        status, out, _ = self.request("POST", "/nothing", self.good)
        self.assertEqual(status, 404)
        status, out, _ = self.request("GET", "/screen")
        self.assertEqual(status, 404)

    def test_still_alive(self):
        self.test_not_an_image()
        status, out, _ = self.request("GET", "/health")
        self.assertIn(status, (200, 503))
        self.assertIn("models", out)


class BusyQueue(ServerCase):
    """At most MAX_WAITING requests wait for the slot; the next one gets 503 busy at once."""

    def test_too_many_waiting(self):
        small = jpeg(Image.new("RGB", (64, 64), (120, 120, 120)))
        body = frames({"boxes": [[0, 0, 10, 10]]}, small)
        srv = self.server
        statuses = []
        srv.slots.acquire()  # stands for a long computation holding the only slot
        try:
            threads = [threading.Thread(target=lambda: statuses.append(self.request("POST", "/redact", body)[0])) for _ in range(srv.max_waiting)]
            for t in threads:
                t.start()
            deadline = time.monotonic() + 30
            while srv.waiting < srv.max_waiting and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(srv.waiting, srv.max_waiting)
            started = time.perf_counter()
            status, out, _ = self.request("POST", "/redact", body)
            self.assertEqual(status, 503, out)
            self.assertEqual(out["error"]["code"], "busy")
            self.assertLess(time.perf_counter() - started, 2.0)
        finally:
            srv.slots.release()
        for t in threads:
            t.join(60)
        self.assertEqual(statuses, [200] * srv.max_waiting)


class MissingModels(ServerCase):
    """Without model files the routes that need them answer 503, never a result."""

    tmp = tempfile.TemporaryDirectory()
    models_dir = tmp.name

    def test_unavailable(self):
        good = jpeg(Image.open(io.BytesIO(load("abo-0007ef0a"))))
        for path in ("/screen", "/fingerprint"):
            status, out, _ = self.post(path, {}, good)
            self.assertEqual(status, 503, out)
            self.assertEqual(out["error"]["code"], "model_unavailable")
        status, out, _ = self.request("GET", "/health")
        self.assertEqual(status, 503)
        self.assertEqual(out["models"]["siglip"], "error")
        # /verify and /redact need no model file.
        status, out, _ = self.post("/verify", {}, good, good)
        self.assertEqual(status, 200)
        self.assertTrue(os.path.isdir(self.models_dir))


if __name__ == "__main__":
    unittest.main()
