"""PDQ against ThreatExchange's pdq/data (plan phase 3, verification 1).

Rule of the PDQ README ("Hashing", rule 2): hashes with quality >= 80 are within distance <= 10
of the C++ reference implementation. Expected values: tests/data/pdq-expected.json.
"""
import json
import os
import unittest

import pdqhash

from app.fingerprint import PDQ_MAX_SIDE, bits_to_hex, pdq
from app.images import decode, to_rgb_array

from .common import hamming, load
from .fixtures import fixture_path, load_fixtures

COMPLIANCE_QUALITY = 80
COMPLIANCE_DISTANCE = 10
# The service hashes a copy of at most PDQ_MAX_SIDE px; on the large test photos this moves the
# hash by at most this many bits (measured: see the test's output), far under the match rule (<= 31).
SHRINK_DISTANCE = 10
HERE = os.path.dirname(os.path.abspath(__file__))


class PdqCompliance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "data", "pdq-expected.json")) as f:
            cls.expected = json.load(f)
        cls.images = {e["pdq_path"]: e for e in load_fixtures() if e["kind"] == "pdq"}

    def rgb(self, path):
        with open(fixture_path(self.images[path]), "rb") as f:
            return to_rgb_array(decode(f.read()))

    def test_reference_hashes(self):
        checked = 0
        for path, exp in self.expected["hashes"].items():
            vec, quality = pdqhash.compute(self.rgb(path))
            d = hamming(bits_to_hex(vec), exp["hash"])
            print(f"  pdq {path}: quality {quality}, distance {d} to the {exp['from']} reference")
            if quality >= COMPLIANCE_QUALITY:
                self.assertLessEqual(d, COMPLIANCE_DISTANCE, path)
                checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_dihedral_order(self):
        path, want = next(iter(self.expected["dihedral"].items()))
        vecs, quality = pdqhash.compute_dihedral(self.rgb(path))
        self.assertGreaterEqual(quality, COMPLIANCE_QUALITY)
        for i, (v, w) in enumerate(zip(vecs, want)):
            self.assertLessEqual(hamming(bits_to_hex(v), w), COMPLIANCE_DISTANCE, f"orientation {i}")

    def test_service_shrink(self):
        large = [e for e in load_fixtures() if e["kind"] in ("photo", "face") and e["id"].startswith("wm-")]
        worst = 0
        for e in large:
            im = decode(load(e["id"]))
            self.assertGreater(max(im.size), PDQ_MAX_SIDE)
            full = bits_to_hex(pdqhash.compute(to_rgb_array(im))[0])
            served = pdq(im)[0][0]
            worst = max(worst, hamming(full, served))
        print(f"  pdq: shrinking to {PDQ_MAX_SIDE} px moves the hash by at most {worst} bits on {len(large)} photos")
        self.assertLessEqual(worst, SHRINK_DISTANCE)


if __name__ == "__main__":
    unittest.main()
