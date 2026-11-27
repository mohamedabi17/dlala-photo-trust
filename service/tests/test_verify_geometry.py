"""/verify's proof (phase 4b): a homography that crushes or folds the picture counts no inliers, and an
inlier counts once per distinct point. No model is needed."""
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import ImageOps

from app import verify as V
from app.images import decode
from app.verify import TRY_MIRROR_BELOW, _distinct, _match_features, _reflects, _sane, verify

from .fixtures import fixture_by_id

SHAPE = (600, 800)  # height, width of the first picture
COPY_INLIERS = 12


class Geometry(unittest.TestCase):
    def test_sane_homographies(self):
        self.assertTrue(_sane(np.eye(3), SHAPE))
        self.assertTrue(_sane(np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1.0]]), SHAPE))  # a shrunk copy
        self.assertTrue(_sane(np.array([[-1.0, 0, 800], [0, 1, 0], [0, 0, 1]]), SHAPE))  # mirrored
        c, s = np.cos(np.radians(15)), np.sin(np.radians(15))
        self.assertTrue(_sane(np.array([[c, -s, 100], [s, c, 0], [0, 0, 1.0]]), SHAPE))  # rotated

    def test_crushed_or_folded_homographies(self):
        self.assertFalse(_sane(np.array([[1e-4, 0, 500], [0, 1e-4, 300], [0, 0, 1.0]]), SHAPE))  # all to one spot
        self.assertFalse(_sane(np.array([[30.0, 0, 0], [0, 30.0, 0], [0, 0, 1]]), SHAPE))  # 30 times larger
        self.assertFalse(_sane(np.array([[1.0, 0, 0], [0, 1, 0], [-0.002, 0, 1]]), SHAPE))  # the far side folds over
        self.assertFalse(_sane(None, SHAPE))

    def test_strong_perspective_area(self):
        # Its 2 x 2 part keeps areas (determinant 1) and the quadrilateral stays convex, but its far side is
        # thrown 90 times away: the true area ratio is about 7,900.
        H = np.array([[1.0, 0, 0], [0, 1, 0], [-0.00124, 0, 1]])
        self.assertAlmostEqual(np.linalg.det(H[:2, :2]), 1.0)
        self.assertFalse(_sane(H, SHAPE))
        self.assertTrue(_sane(np.array([[1.0, 0, 0], [0, 1, 0], [-0.0011, 0, 1]]), SHAPE))  # about 39 times: kept

    def test_rejected_homography_is_not_returned(self):
        # 40 matches from all over the first picture onto one 2 px spot of the second: RANSAC fits them all.
        rng = np.random.default_rng(0)
        pa = rng.uniform([0, 0], [800, 600], (40, 2))
        pb = np.array([400, 300]) + rng.uniform(-1, 1, (40, 2))
        _, raw = cv2.findHomography(pa.reshape(-1, 1, 2).astype(np.float32), pb.reshape(-1, 1, 2).astype(np.float32), cv2.RANSAC, 5.0)
        self.assertEqual(int(raw.sum()), 40)
        da = (rng.normal(size=(40, 128)) * 50).astype(np.float32)
        fa = {"kp": [cv2.KeyPoint(float(x), float(y), 5) for x, y in pa], "des": da, "shape": SHAPE}
        fb = {"kp": [cv2.KeyPoint(float(x), float(y), 5) for x, y in pb], "des": da + 0.1, "shape": SHAPE}
        res = _match_features(fa, fb)
        self.assertEqual(len(res["good"]), 40)
        self.assertEqual(res["inliers"], 0)
        self.assertIsNone(res["H"])
        self.assertIsNone(res["mask"])  # no line drawn on the evidence picture

    def test_reflection_found_unmirrored_tries_the_mirror(self):
        a = decode(open(fixture_by_id("pdq-bridge-1-original"), "rb").read())
        flip = np.array([[-1.0, 0, 800], [0, 1, 0], [0, 0, 1]])
        first = {"ka": [], "kb": [], "good": [], "mask": None, "H": flip, "inliers": TRY_MIRROR_BELOW + 20}
        better = {"ka": [], "kb": [], "good": [], "mask": None, "H": np.eye(3), "inliers": 400}
        worse = dict(better, inliers=10)
        with mock.patch.object(V, "_match_features", side_effect=[first, better]) as m:
            out = verify(a, a, {})
        self.assertEqual(m.call_count, 2)  # tried although above TRY_MIRROR_BELOW
        self.assertEqual(out["inliers"], 400)
        self.assertTrue(out["mirrored"])
        with mock.patch.object(V, "_match_features", side_effect=[first, worse]):
            out = verify(a, a, {})
        self.assertEqual(out["inliers"], TRY_MIRROR_BELOW + 20)
        self.assertTrue(out["mirrored"])  # the kept homography turns the picture over
        self.assertTrue(_reflects(flip))
        self.assertFalse(_reflects(np.eye(3)))

    def test_distinct_points(self):
        self.assertEqual(_distinct([(10.2, 10.1), (10.9, 10.8), (10.0, 10.0), (50, 50)]), 2)

    def test_copy_and_mirror_found_other_photo_not(self):
        a = decode(open(fixture_by_id("pdq-bridge-1-original"), "rb").read())
        b = decode(open(fixture_by_id("wm-33298044"), "rb").read())
        self.assertGreaterEqual(verify(a, a.resize((800, 502)), {})["inliers"], 100)
        m = verify(a, ImageOps.mirror(a), {})
        self.assertTrue(m["mirrored"])
        self.assertGreaterEqual(m["inliers"], TRY_MIRROR_BELOW)
        self.assertLess(verify(a, b, {})["inliers"], COPY_INLIERS)
        print(f"  other photo: {verify(a, b, {})['inliers']} inliers")


if __name__ == "__main__":
    unittest.main()
