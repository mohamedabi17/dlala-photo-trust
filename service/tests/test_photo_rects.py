"""The photo rectangle finder (app/screen.py photo_rects_part, phase 4b): a photo on a white, flat or
dark page, the cells of a collage with or without gutters, an ordinary photo left whole, and its time.
No model is needed: the finder is OpenCV and NumPy only."""
import io
import statistics
import time
import tracemalloc
import unittest

import cv2
import numpy as np
from PIL import Image

from app.images import decode
from app.screen import photo_rects_part

from .fixtures import fixture_by_id

# Measured on the calibrate half of calibration-fingerprints-2026-11-25.md: a found rectangle covers
# the photo with an intersection over union of at least 0.7 on 88 % of the synthetic pages and 82 % of
# the collages. Here, on clean cases, each must.
RECT_IOU = 0.7
# The plan's budget for the finder is 150 ms at 1600 px on one thread. Wall time depends on the machine and
# its load, so the test bounds ratios of CPU time: the finder at 1600 px against a fixed reference workload
# of the same kinds of operations (resize, Canny, box filter, prefix sums), measured 1.8 to 2.7 times it on
# the work laptop (reference 18 to 28 ms; the budget is about 7 times it); and the finder at 4096 px against
# 1600 px, measured 1.3 to 1.5 (3 to 4 with a full-size copy of the original, before 26/11).
MAX_COST_VS_REFERENCE = 5
MAX_COST_4096_VS_1600 = 2.0
# NumPy/OpenCV arrays held at 4096 px beyond those at 1600 px (measured 0; a full-size copy of a
# 4096 x 3072 original alone is 38 MB).
MAX_EXTRA_PEAK_MB_4096 = 5
PHOTOS = ("pdq-bridge-1-original", "wm-sally-ride", "wm-mae-jemison", "wm-35195774")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def photo(fid, width):
    im = Image.open(fixture_by_id(fid)).convert("RGB")
    return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)


def as_upload(canvas):
    """JPEG quality 90 and back through the service's decoder, as an upload would be."""
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="JPEG", quality=90)
    return decode(buf.getvalue())


def page(background, width=720, height=1480, photo_id="pdq-bridge-1-original", margin=24):
    """A phone-sized listing page: status bar, header, the photo, a title, text lines and a button."""
    ink = (230, 230, 230) if sum(background) < 300 else (25, 25, 30)
    c = np.full((height, width, 3), background, np.uint8)
    cv2.putText(c, "14:32", (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ink, 2, cv2.LINE_AA)
    cv2.putText(c, "87 %", (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ink, 2, cv2.LINE_AA)
    c[44:120] = (31, 95, 139)
    cv2.putText(c, "annonces.example", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    p = np.asarray(photo(photo_id, width - 2 * margin))
    top = 150
    c[top:top + p.shape[0], margin:margin + p.shape[1]] = p
    box = [margin, top, margin + p.shape[1], top + p.shape[0]]
    y = box[3] + 60
    cv2.putText(c, "Pont ancien, tirage photo", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, ink, 2, cv2.LINE_AA)
    cv2.putText(c, "45 EUR", (24, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, ink, 3, cv2.LINE_AA)
    for k in range(6):
        cv2.putText(c, "tres bon etat, envoi possible, remise en main propre", (24, y + 120 + 36 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink, 1, cv2.LINE_AA)
    cv2.rectangle(c, (24, height - 120), (300, height - 64), (255, 110, 20), -1)
    return c, box


def collage(gutter, colour=(255, 255, 255), cell=(560, 420)):
    cw, ch = cell
    w, h = 2 * cw + 3 * gutter, 2 * ch + 3 * gutter
    c = np.full((h, w, 3), colour, np.uint8)
    boxes = []
    for k, fid in enumerate(PHOTOS):
        x, y = gutter + (k % 2) * (cw + gutter), gutter + (k // 2) * (ch + gutter)
        im = Image.open(fixture_by_id(fid)).convert("RGB")
        r = max(cw / im.width, ch / im.height)  # fill the cell, centred (what collage apps do)
        im = im.resize((max(cw, round(im.width * r)), max(ch, round(im.height * r))), Image.Resampling.LANCZOS)
        ox, oy = (im.width - cw) // 2, (im.height - ch) // 2
        c[y:y + ch, x:x + cw] = np.asarray(im)[oy:oy + ch, ox:ox + cw]
        boxes.append([x, y, x + cw, y + ch])
    return c, boxes


class PhotoRects(unittest.TestCase):
    def best(self, rects, box):
        return max([iou(r["box"], box) for r in rects], default=0.0)

    def test_photo_on_white_grey_and_dark_pages(self):
        for bg in ((255, 255, 255), (243, 244, 246), (18, 18, 20), (250, 236, 220)):
            canvas, box = page(bg)
            rects, flat, cells = photo_rects_part(as_upload(canvas))
            print(f"  page {bg}: {[r['box'] for r in rects]}, flat {flat:.2f}")
            self.assertGreaterEqual(self.best(rects, box), RECT_IOU, bg)
            self.assertEqual(cells, 0)

    def test_full_width_photo_with_its_own_dark_border(self):
        canvas, box = page((243, 244, 246), photo_id="wm-33285967", margin=0)
        rects, _, _ = photo_rects_part(as_upload(canvas))
        self.assertGreaterEqual(self.best(rects, box), RECT_IOU)

    def test_collage_cells(self):
        for gutter, colour in ((0, (255, 255, 255)), (10, (255, 255, 255)), (8, (0, 0, 0))):
            canvas, boxes = collage(gutter, colour)
            rects, _, cells = photo_rects_part(as_upload(canvas))
            print(f"  collage gutter {gutter}: {[r['box'] for r in rects]}, cells {cells}")
            for b in boxes:
                self.assertGreaterEqual(self.best(rects, b), RECT_IOU, (gutter, b))
            self.assertEqual(cells, 4)

    def test_ocr_boxes_are_accepted(self):
        canvas, box = page((255, 255, 255))
        im = as_upload(canvas)
        ocr = [{"text": "annonces.example", "box": [20, 70, 330, 100]}, [24, box[3] + 30, 600, box[3] + 70]]
        rects, _, _ = photo_rects_part(im, ocr)
        self.assertGreaterEqual(self.best(rects, box), RECT_IOU)

    def test_ordinary_photo_stays_whole(self):
        for fid in ("pdq-bridge-1-original", "wm-33298044"):
            im = decode(open(fixture_by_id(fid), "rb").read())
            rects, _, cells = photo_rects_part(im)
            self.assertEqual(cells, 0, fid)

    def test_long_thin_image(self):
        for size in ((8192, 16), (16, 8192), (6000, 40)):
            rects, _, cells = photo_rects_part(Image.new("RGB", size, (90, 160, 30)))
            self.assertEqual((rects, cells), ([], 0), size)

    def test_cost(self):
        """Machine-independent: CPU time against a fixed reference workload of the same kinds of operations,
        and the 4096 px original against the 1600 px one (it must not copy the original at full size)."""
        cv2.setNumThreads(1)
        rng = np.random.default_rng(0)
        ref_in = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)

        def reference():
            small = cv2.resize(ref_in, (800, 600), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            cv2.Canny(g, 50, 150)
            cv2.boxFilter(g.astype(np.float32), -1, (7, 7))
            np.cumsum(small.astype(np.float64), axis=0)

        def cpu_ms(fn, runs=5):
            fn()
            ms = []
            for _ in range(runs):
                t = time.process_time()
                fn()
                ms.append((time.process_time() - t) * 1000)
            return statistics.median(ms)

        im1600 = as_upload(collage(0, cell=(800, 600))[0])  # 1600 x 1200
        im4096 = as_upload(collage(0, cell=(2048, 1536))[0])  # 4096 x 3072
        ref, t1600, t4096 = cpu_ms(reference), cpu_ms(lambda: photo_rects_part(im1600)), cpu_ms(lambda: photo_rects_part(im4096))
        peaks = []
        for im in (im1600, im4096):
            tracemalloc.start()
            photo_rects_part(im)
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        print(f"  reference {ref:.1f} ms; 1600 px {t1600:.0f} ms ({t1600 / ref:.1f} x); 4096 px {t4096:.0f} ms "
              f"({t4096 / t1600:.2f} x the 1600 px); peak of NumPy/OpenCV arrays {peaks[0]:.1f} MB at 1600 px, {peaks[1]:.1f} MB at 4096 px")
        self.assertLess(t1600 / ref, MAX_COST_VS_REFERENCE)
        self.assertLess(t4096 / t1600, MAX_COST_4096_VS_1600)
        self.assertLess(peaks[1] - peaks[0], MAX_EXTRA_PEAK_MB_4096)


if __name__ == "__main__":
    unittest.main()
