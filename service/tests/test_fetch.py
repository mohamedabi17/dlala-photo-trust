"""The model fetcher refuses addresses that are not https, before downloading anything."""
import json
import os
import tempfile
import unittest

from app import fetch_models


class FetchRefusesPlainHttp(unittest.TestCase):
    def test_http_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "lock.json")
            with open(lock, "w") as f:
                json.dump({"files": [{"id": "x", "group": "vision", "path": "vision/x.onnx", "url": "http://example.invalid/x.onnx",
                                      "sha256": "0" * 64, "size": 1, "licence": "MIT", "source_page": "http://example.invalid"}]}, f)
            dest = os.path.join(tmp, "models")
            self.assertEqual(fetch_models.main(["--lock", lock, "--dest", dest, "--groups", "vision"]), 2)
            self.assertFalse(os.path.exists(os.path.join(dest, "vision")))


if __name__ == "__main__":
    unittest.main()
