"""/redact makes a face undetectable by YuNet (plan phase 3, verification 3)."""
import io
import json
import unittest

from PIL import Image

from app.images import decode
from app.models import YUNET_SCORE_FLOOR
from app.redact import EXPAND

from .common import ServerCase, load

FACE_FIXTURES = ("wm-sally-ride", "wm-mae-jemison")
# A face found again on the blurred photo must not overlap the old box by more than this.
MAX_IOU_AFTER = 0.1


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class RedactHidesFaces(ServerCase):
    def faces(self, data):
        status, out, _ = self.post("/screen", {"parts": ["faces"]}, data)
        self.assertEqual(status, 200, out)
        return out["faces"]

    def test_faces_are_gone(self):
        for fid in FACE_FIXTURES:
            data = load(fid)
            before = self.faces(data)
            self.assertTrue(before, f"YuNet finds no face on {fid}")
            boxes = [f["box"] for f in before]
            status, jpeg_out, r = self.post("/redact", {"boxes": boxes}, data)
            self.assertEqual(status, 200)
            self.assertEqual(r.getheader("Content-Type"), "image/jpeg")
            done = json.loads(r.getheader("X-Redacted-Boxes"))
            for b, d in zip(boxes, done):  # each box grew by EXPAND around its centre
                self.assertAlmostEqual((d[2] - d[0]) / (b[2] - b[0]), 1 + EXPAND, delta=0.02)
            im = Image.open(io.BytesIO(jpeg_out))
            self.assertEqual(im.size, decode(data).size)
            after = self.faces(jpeg_out)
            left = [f for f in after for b in boxes if iou(f["box"], b) > MAX_IOU_AFTER]
            print(f"  redact {fid}: {len(before)} face(s) found (score {before[0]['score']}), {len(left)} after blurring (floor {YUNET_SCORE_FLOOR})")
            self.assertEqual(left, [], fid)


if __name__ == "__main__":
    unittest.main()
