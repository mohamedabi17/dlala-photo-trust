"""Helpers shared by the tests: an in-process server on a free port, request framing, images."""
import base64
import http.client
import io
import json
import os
import struct
import threading
import unittest

import numpy as np
from PIL import Image

from app.models import Registry
from app.server import make_server

from .fixtures import fixture_by_id

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")


def frames(meta, *images):
    """The application/x-dlala-frames body: 4-byte big-endian JSON length, JSON, then the images."""
    meta = dict(meta, lengths=[len(i) for i in images])
    head = json.dumps(meta).encode()
    return struct.pack(">I", len(head)) + head + b"".join(images)


def jpeg(im, quality=90):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def load(fid):
    with open(fixture_by_id(fid), "rb") as f:
        return f.read()


def pil(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def fp16(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


class ServerCase(unittest.TestCase):
    """Starts one server per test class, on 127.0.0.1 and a free port."""

    models_dir = MODELS_DIR
    idle_unload_s = 0  # off unless a test class turns it on

    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, Registry(models_dir=cls.models_dir), cls.idle_unload_s)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=b"", ctype="application/x-dlala-frames", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        h = {"Content-Type": ctype}
        h.update(headers or {})
        if method == "POST" and "Content-Length" not in h:
            h["Content-Length"] = str(len(body))
        conn.putrequest(method, path)
        for k, v in h.items():
            conn.putheader(k, v)
        conn.endheaders()
        if body:
            try:
                conn.send(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        r = conn.getresponse()
        data = r.read()
        conn.close()
        ctype_out = r.getheader("Content-Type", "")
        return r.status, (json.loads(data) if ctype_out.startswith("application/json") else data), r

    def post(self, path, meta, *images):
        return self.request("POST", path, frames(meta, *images))
