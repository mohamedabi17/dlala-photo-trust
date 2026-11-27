"""ONNX against the PyTorch reference (plan phase 3, verification 2): cosine >= 0.999 on 20 images.

The reference vectors were computed with transformers by tools/make-reference.py
(tests/data/reference.json says with which versions). DINOv2-base is checked only when its file
was fetched (group vision-dino-base).
"""
import json
import os
import unittest

import numpy as np

from app.images import decode
from app.models import Registry

from .common import MODELS_DIR, load

MIN_COSINE = 0.999
HERE = os.path.dirname(os.path.abspath(__file__))


class OnnxMatchesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "data", "reference.json")) as f:
            cls.meta = json.load(f)
        cls.images = [decode(load(fid)) for fid in cls.meta["fixtures"]]
        assert len(cls.images) == 20

    def check(self, name, vectors):
        ref = np.load(os.path.join(HERE, "data", f"reference-{name}.npy"))
        cos = (vectors * ref).sum(axis=1)
        print(f"  {name}: cosine to PyTorch min {cos.min():.6f}, mean {cos.mean():.6f} on {len(cos)} images")
        self.assertGreaterEqual(float(cos.min()), MIN_COSINE)

    def test_siglip(self):
        self.check("siglip", Registry(models_dir=MODELS_DIR).siglip_embed(self.images))

    def test_dino_small(self):
        self.check("dino-small", Registry(models_dir=MODELS_DIR, dino_variant="small").dino_embed(self.images))

    def test_dino_base(self):
        reg = Registry(models_dir=MODELS_DIR, dino_variant="base")
        if reg.file_state("dino-base") != "present":
            self.skipTest("DINOv2-base not fetched (python -m app.fetch_models --groups vision-dino-base)")
        self.check("dino-base", reg.dino_embed(self.images))


if __name__ == "__main__":
    unittest.main()
