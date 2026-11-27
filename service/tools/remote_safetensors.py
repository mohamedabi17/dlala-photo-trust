"""Read chosen tensors of a remote .safetensors file with HTTP range requests, into memory only.

Work-computer tool (tools/build-prompts.py, tools/make-reference.py): the full SigLIP 2 file is
1.5 GB, its text embedding table alone 786 MB, and only a few hundred of its rows are needed.
Format: 8-byte little-endian header length, a JSON header {name: {dtype, shape, data_offsets}},
then the raw data (https://huggingface.co/docs/safetensors/index, "Format").
"""
import json
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
PARALLEL_REQUESTS = 8
TIMEOUT_S = 120
USER_AGENT = "dlala-vision-tools/1"


class RemoteSafetensors:
    def __init__(self, url):
        # Resolve the redirect once (Hugging Face sends file bytes from a CDN address).
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7", "User-Agent": USER_AGENT}), timeout=TIMEOUT_S)
        self.url = r.geturl()
        n = struct.unpack("<Q", r.read())[0]
        self.header = json.loads(self._range(8, 8 + n - 1))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def _range(self, a, b):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = r.read()
        if len(data) != b - a + 1:
            raise IOError(f"short read {len(data)} for range {a}-{b}")
        return data

    def names(self, prefix=""):
        return [k for k in self.header if k.startswith(prefix)]

    def tensor(self, name):
        h = self.header[name]
        s, e = h["data_offsets"]
        buf = self._range(self.data_start + s, self.data_start + e - 1)
        return np.frombuffer(buf, dtype=DTYPES[h["dtype"]]).reshape(h["shape"]).copy()

    def tensors(self, names):
        with ThreadPoolExecutor(PARALLEL_REQUESTS) as ex:
            return dict(zip(names, ex.map(self.tensor, names)))

    def rows(self, name, ids):
        """Rows ids of a 2-D tensor, in that order."""
        h = self.header[name]
        dt = np.dtype(DTYPES[h["dtype"]])
        width = h["shape"][1]
        row_bytes = width * dt.itemsize
        s = self.data_start + h["data_offsets"][0]

        def one(i):
            return np.frombuffer(self._range(s + i * row_bytes, s + (i + 1) * row_bytes - 1), dtype=dt)

        with ThreadPoolExecutor(PARALLEL_REQUESTS) as ex:
            return np.stack(list(ex.map(one, ids)))
