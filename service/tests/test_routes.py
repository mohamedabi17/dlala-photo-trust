"""What each route returns on good input (shapes, and that the signals make sense)."""
import base64
import io
import time
import unittest

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.screen import RESTRICTED_GROUPS

from .common import ServerCase, fp16, hamming, jpeg, load, pil

# PDQ match rule of the ThreatExchange README (plan phase 4).
PDQ_MATCH = 31
# Inliers that prove a copy in the plan's benchmark (phase 4); two different objects stay far below.
COPY_INLIERS = 12
# A synthetic phone screenshot (iPhone 12-15 size) with a photo pasted in it.
PHONE = (1170, 2532)
PASTE_BOX = (85, 640, 1085, 1640)
RECT_IOU = 0.8


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def screenshot():
    canvas = np.full((PHONE[1], PHONE[0], 3), 245, np.uint8)
    cv2.putText(canvas, "12:34", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(canvas, "87%", (930, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(canvas, "Netsuke ancien", (85, 520), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (10, 10, 10), 6, cv2.LINE_AA)
    cv2.putText(canvas, "Ajouter au panier", (85, 1900), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (10, 10, 10), 6, cv2.LINE_AA)
    photo = pil(load("wm-33284915")).resize((PASTE_BOX[2] - PASTE_BOX[0], PASTE_BOX[3] - PASTE_BOX[1]))
    canvas[PASTE_BOX[1]:PASTE_BOX[3], PASTE_BOX[0]:PASTE_BOX[2]] = np.asarray(photo)
    return jpeg(Image.fromarray(canvas), 92)


class Routes(ServerCase):
    def ok(self, path, meta, *images):
        t = time.perf_counter()
        status, out, r = self.post(path, meta, *images)
        self.assertEqual(status, 200, out)
        print(f"  {path} {meta or ''}: {(time.perf_counter() - t) * 1000:.0f} ms")
        return out

    def test_health(self):
        status, out, _ = self.request("GET", "/health")
        self.assertEqual(status, 200, out)
        self.assertEqual(out["api_version"], "vision-v1")
        self.assertEqual(out["models"]["probe"], "absent")

    def test_fingerprint(self):
        data = load("wm-sally-ride")
        out = self.ok("/fingerprint", {"regions": [[1000, 700, 1900, 1800]]}, data)
        self.assertEqual(len(out["pdq"]), 8)
        self.assertTrue(all(len(h) == 64 for h in out["pdq"]))
        self.assertGreater(out["quality"], 49)
        v = fp16(out["dino"])
        self.assertEqual(len(v), out["dino_dim"])
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, delta=0.01)
        self.assertEqual(out["regions"][0]["box"], [1000, 700, 1900, 1800])
        self.assertEqual(len(out["regions"][0]["pdq"]), 8)
        # A mirrored, recompressed copy is found among the 8 orientations of the original.
        copy = jpeg(ImageOps.mirror(pil(data)).resize((900, 1125)), 60)
        c = self.ok("/fingerprint", {}, copy)
        self.assertLessEqual(min(hamming(c["pdq"][0], h) for h in out["pdq"]), PDQ_MATCH)
        cos = float(fp16(c["dino"]) @ v)
        print(f"  mirrored copy: DINOv2 cosine {cos:.3f}")

    def test_fingerprint_tiles(self):
        data = load("wm-sally-ride")
        w, h = pil(data).size
        out = self.ok("/fingerprint", {"regions": [[1000, 700, 1900, 1800]], "tiles": True}, data)
        self.assertEqual([r["tile"] for r in out["regions"]], [False] + [True] * 5)
        self.assertEqual([r["box"] for r in out["regions"][1:]], [[0, 0, w // 2, h // 2], [w // 2, 0, w, h // 2], [0, h // 2, w // 2, h],
                                                                  [w // 2, h // 2, w, h], [w // 4, h // 4, w - w // 4, h - h // 4]])
        # the quadrant holding the face is nearer the face region than the whole photo is
        face = fp16(out["regions"][0]["dino"])
        self.assertTrue(all(len(r["pdq"]) == 8 for r in out["regions"]))
        self.assertGreater(max(float(fp16(r["dino"]) @ face) for r in out["regions"][1:]), 0.5)
        status, err, _ = self.post("/fingerprint", {"tiles": "yes"}, data)
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "bad_meta")

    def test_verify(self):
        a = load("wm-33284613")
        im = pil(a)
        w, h = im.size
        crop = im.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))).resize((800, 800))
        out = self.ok("/verify", {}, a, jpeg(crop, 50))
        print(f"  verify copy: {out['inliers']} inliers of {out['matches']} matches")
        self.assertGreaterEqual(out["inliers"], COPY_INLIERS)
        self.assertFalse(out["mirrored"])
        Image.open(io.BytesIO(base64.b64decode(out["evidence_jpeg"]))).verify()
        mir = self.ok("/verify", {}, a, jpeg(ImageOps.mirror(crop), 70))
        print(f"  verify mirrored copy: {mir['inliers']} inliers, mirrored={mir['mirrored']}")
        self.assertGreaterEqual(mir["inliers"], COPY_INLIERS)
        self.assertTrue(mir["mirrored"])
        other = self.ok("/verify", {}, a, load("wm-33298044"))
        print(f"  verify two different netsuke: {other['inliers']} inliers")
        self.assertLess(other["inliers"], COPY_INLIERS)

    def test_screen_face_and_text(self):
        out = self.ok("/screen", {"parts": ["siglip", "hotspots", "ocr", "faces", "photo_rects"]}, load("wm-sally-ride"))
        self.assertTrue(out["faces"])
        self.assertIn("NASA", " ".join(o["text"] for o in out["ocr"]))
        self.assertEqual(set(out["siglip"]), set(RESTRICTED_GROUPS))
        self.assertIsNone(out["probe"])
        self.assertEqual(len(out["hotspots"]), 4)
        self.assertEqual(len(fp16(out["siglip_vector"])), 768)
        print(f"  timings {out['timings_ms']}")

    def test_screen_ivory(self):
        out = self.ok("/screen", {"parts": ["siglip", "hotspots"]}, load("wm-33284915"))
        top = max(out["siglip"], key=out["siglip"].get)
        self.assertEqual(top, "ivory")
        self.assertGreater(out["siglip"]["ivory"], out["siglip_ordinary"])
        self.assertEqual(max(out["hotspots"][0]["scores"], key=out["hotspots"][0]["scores"].get), "ivory")

    def test_screen_screenshot(self):
        out = self.ok("/screen", {}, screenshot())
        signs = out["screenshot_signs"]
        print(f"  screenshot signs {signs}; rects {out['photo_rects']}")
        self.assertTrue(signs["phone_aspect"])
        self.assertTrue(signs["screen_size"])
        self.assertTrue(signs["status_bar"])
        self.assertGreaterEqual(iou(out["photo_rects"][0]["box"], PASTE_BOX), RECT_IOU)
        self.assertIn("panier", " ".join(o["text"] for o in out["ocr"]).lower())
        self.assertNotIn("hotspots", out)

    def test_ordinary_photo_has_no_rect(self):
        out = self.ok("/screen", {"parts": ["photo_rects"]}, load("wm-33298044"))
        self.assertEqual(out["photo_rects"], [])
        self.assertFalse(out["screenshot_signs"]["phone_aspect"])

    def test_largest_original(self):
        im = pil(load("wm-sally-ride"))
        big = im.resize((round(im.width * 4096 / im.height), 4096))
        out = self.ok("/screen", {}, jpeg(big, 90))
        self.assertEqual(out["size"], list(big.size))
        self.assertTrue(out["faces"])


if __name__ == "__main__":
    unittest.main()
