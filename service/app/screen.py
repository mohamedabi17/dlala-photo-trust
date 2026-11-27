"""POST /screen: the fast filters of plan phase 5, step A. Raw scores only: every threshold that
turns a score into a decision lives in the API's config/photo-checks.json (phase 1), not here."""
import re
import time

import cv2
import numpy as np

from .fingerprint import fp16_b64
from .images import BadInput, shrink, to_bgr_array
from .models import OCR_MAX_SIDE

PARTS = ("siglip", "hotspots", "ocr", "faces", "photo_rects")
# Hotspots cost one SigLIP pass per tile (4 more by default): only on request, for the photos that
# go to the small model (plan phase 5, step B), not on every photo.
DEFAULT_PARTS = ("siglip", "ocr", "faces", "photo_rects")
RESTRICTED_GROUPS = ("ivory", "protected_species", "weapons", "counterfeit", "cultural_goods", "human_remains", "nazi")
NEGATIVE_GROUP = "ordinary"
# Prompts listed per photo in siglip_top (for the review page and calibration).
TOP_PROMPTS = 3

# Hotspots: the image cut into a grid of overlapping tiles, each scored by SigLIP 2 at full
# resolution of the original (the "zoom" of the cascade). 2 gives 4 tiles; 0 turns them off.
DEFAULT_TILE_GRID = 2
MAX_TILE_GRID = 3
# Each tile is this much larger than 1/grid of the side, so an object on a cut is whole in one tile.
TILE_OVERLAP = 0.25

# Faces: YuNet runs on a copy at most this large (a face of 30 px on a 4096 px original stays >= 9 px).
FACE_MAX_SIDE = 1280

# Photo rectangles: computed on a copy at most this large.
RECT_MAX_SIDE = 800
# A pixel is "flat" (screen background, UI panel) when the grey-level spread around it is below this.
FLAT_WINDOW = 7
FLAT_STD = 3.0
# Gaps closed inside a photo, and strokes removed (text lines), as a fraction of the shorter side.
CLOSE_FRACTION = 0.01
OPEN_FRACTION = 0.03
# A candidate covers between these fractions of the image (the whole image is not "a photo inside").
MIN_RECT_FRACTION = 0.03
MAX_RECT_FRACTION = 0.92
# How rectangular a candidate must be (its area over its bounding box), and its side ratio.
MIN_RECT_FILL = 0.85
MAX_RECT_ASPECT = 4.0
# Natural photos use many colours; UI blocks and text use few (colours counted on 4 bits a channel).
NATURAL_COLORS = 256
COLOR_BITS_DROPPED = 4
NATURAL_SUBSAMPLE_PIXELS = 40000
# Edge rectangles: polygon approximation tolerance (fraction of the perimeter).
POLY_EPSILON = 0.02
# Candidates overlapping more than this (intersection over union) are the same rectangle.
RECT_IOU_SAME = 0.6
MAX_PHOTO_RECTS = 4
# Phase 4b (calibration-fingerprints-2026-11-25.md): photos on white or flat pages, and collages.
# A textured blob that is not a full rectangle (a photo whose sky or white wall reaches its edge) still
# counts when at least MIN_SIDES_SEEN sides of its box are lined with photo along MIN_SIDE_CONTACT of
# their length: a photo is cut by straight lines, an object on a flat background only touches its box
# at a few points (ABO's products on white stay out).
MIN_BLOB_FILL = 0.5
MIN_SIDE_CONTACT = 0.35
MIN_SIDES_SEEN = 3
# Px (on the copy) just inside a side where its contact is measured.
SIDE_BAND = 3
# OCR boxes (when /screen read the text) are never photo: they leave the textured mask, grown by this (px of the copy).
OCR_PAD = 2
# The cuts are searched on a copy at most this large (their boxes are scaled back to the RECT_MAX_SIDE copy),
# and only when its shorter side has at least MIN_CUT_SIDE px.
CUT_MAX_SIDE = 400
MIN_CUT_SIDE = 8
# Recursive cuts (the "XY-cut" of page layout analysis): a part is cut along a straight line across it
# where the picture changes at once (two photos of a collage side by side, a full-width photo's edge), or
# along a uniform band (a collage's gutter, the margin between page blocks). Leaves are candidates.
CUT_RANGE = (0.12, 0.88)  # where a cut may lie inside a part, as a share of its side
CUT_MIN_STEP = 24  # colour step across the line, summed over R, G, B (0-765)
# Share of the line's length where the step is the local maximum at that exact place. Chosen on the
# calibrate half of calibration-fingerprints-2026-11-25.md: 0.7 misses 3 % of the seams of borderless
# collages and cuts 4 % of ordinary photos' axes (0.5: 0 % and 15 %).
CUT_MIN_SHARE = 0.7
# Steps are trusted for the first cuts only (a 2 x 2 grid needs two levels); deeper, only uniform bands
# cut (inside a photo, straight steps are its own edges: a table, a shelf, a horizon).
MAX_STEP_DEPTH = 2
UNIFORM_STD = 4.0  # a row or column of a part is uniform when its spread is below this (each channel)
# A uniform band is split where its colour changes by more than this (a photo's own flat border next to
# the page's margin); the band of the page's colour (the commonest flat row colour) is cut first.
BAND_COLOUR_STEP = 8.0
MAX_CUT_DEPTH = 4
MAX_LEAVES = 24
MIN_LEAF_SIDE = 0.12  # a leaf narrower than this share of the image's side is not cut further
# A collage: at least this many leaves of similar size (each at least this share of the image) tile it.
GRID_MIN_CELLS = 2
GRID_MIN_CELL_FRACTION = 0.12
GRID_MIN_COVER = 0.85

# Screenshot signs.
PHONE_ASPECT_RANGE = (1.75, 2.35)  # portrait phone screens, 16:9 to 21:9
STATUS_BAR_FRACTION = 0.06  # top strip of the image where a phone shows the time and battery
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:h][0-5]\d\b")
BATTERY_RE = re.compile(r"\b\d{1,3}\s?%")
# Screen sizes of common phones, tablets and computers (either orientation is matched).
SCREEN_SIZES = {
    (750, 1334), (828, 1792), (1125, 2436), (1170, 2532), (1179, 2556), (1242, 2208), (1242, 2688),
    (1284, 2778), (1290, 2796), (1320, 2868), (1206, 2622), (720, 1600), (1080, 1920), (1080, 2340),
    (1080, 2400), (1080, 2412), (1440, 3088), (1440, 3120), (1440, 3200), (1536, 2048), (1640, 2360),
    (1668, 2388), (2048, 2732), (1366, 768), (1440, 900), (1536, 864), (1920, 1080), (1920, 1200),
    (2560, 1440), (2560, 1600), (2880, 1800), (3024, 1964), (3456, 2234), (3840, 2160),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tile_boxes(w, h, grid):
    if grid <= 0:
        return []
    tw, th = min(w, round(w / grid * (1 + TILE_OVERLAP))), min(h, round(h / grid * (1 + TILE_OVERLAP)))
    xs = np.linspace(0, w - tw, grid).round().astype(int)
    ys = np.linspace(0, h - th, grid).round().astype(int)
    return [[int(x), int(y), int(x) + tw, int(y) + th] for y in ys for x in xs]


def group_scores(probs, groups):
    out = {}
    for g in dict.fromkeys(groups):
        out[g] = float(probs[[i for i, x in enumerate(groups) if x == g]].max())
    return out


def siglip_part(reg, im, grid, want_hotspots):
    w, h = im.size
    boxes = tile_boxes(w, h, grid) if want_hotspots else []
    vecs = reg.siglip_embed([im], boxes)
    P = reg.prompts()
    probs = _sigmoid(vecs @ P["vectors"].T * P["scale"] + P["bias"])
    full = group_scores(probs[0], P["groups"])
    order = np.argsort(-probs[0])[:TOP_PROMPTS]
    out = {
        "siglip": {g: round(full[g], 5) for g in RESTRICTED_GROUPS},
        "siglip_ordinary": round(full.get(NEGATIVE_GROUP, 0.0), 5),
        "siglip_top": [{"group": P["groups"][i], "text": P["meta"]["prompts"][i]["text"], "score": round(float(probs[0][i]), 5)} for i in order],
        "siglip_vector": fp16_b64(vecs[0]),
    }
    probe = reg.probe()
    if probe is None:
        out["probe"] = None
    else:
        p = _sigmoid(probe["w"] @ vecs[0] + probe["b"])
        out["probe"] = {g: round(float(s), 5) for g, s in zip(probe["groups"], p)}
    if want_hotspots:
        spots = []
        for b, pr in zip(boxes, probs[1:]):
            s = group_scores(pr, P["groups"])
            spots.append({"box": b, "scores": {g: round(s[g], 5) for g in RESTRICTED_GROUPS}})
        spots.sort(key=lambda x: -max(x["scores"].values()))
        out["hotspots"] = spots
    return out


def ocr_part(reg, im):
    # Shrunk here rather than inside RapidOCR, so it never holds a copy of the full original.
    small, scale = shrink(im, OCR_MAX_SIDE)
    with reg.ocr_lock:
        res = reg.ocr()(to_bgr_array(small))
    del small
    items = []
    if res is not None and getattr(res, "boxes", None) is not None and res.txts:
        for quad, text, score in zip(res.boxes, res.txts, res.scores):
            q = np.asarray(quad, dtype=float) / scale
            items.append({
                "text": str(text), "score": round(float(score), 4),
                "box": [int(q[:, 0].min()), int(q[:, 1].min()), int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))],
                "quad": [[int(round(x)), int(round(y))] for x, y in q],
            })
    return items


def faces_part(reg, im):
    small, scale = shrink(im, FACE_MAX_SIDE)
    sw, sh = small.size
    with reg.yunet_lock:
        det = reg.yunet()
        det.setInputSize((sw, sh))
        _, faces = det.detect(to_bgr_array(small))
    out = []
    w, h = im.size
    for f in faces if faces is not None else []:
        # On a very thin image (8192 x 16 shrinks to 1280 x 2) YuNet can answer infinite boxes: not a face.
        if not np.all(np.isfinite(f[:15])):
            continue
        x, y, fw, fh = (float(v) / scale for v in f[:4])
        out.append({
            "box": [max(0, int(x)), max(0, int(y)), min(w, int(np.ceil(x + fw))), min(h, int(np.ceil(y + fh)))],
            "landmarks": [[round(float(f[4 + 2 * i]) / scale, 1), round(float(f[5 + 2 * i]) / scale, 1)] for i in range(5)],
            "score": round(float(f[14]), 4),
        })
    out.sort(key=lambda f: -f["score"])
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _sides_seen(mask, box, check=(True, True, True, True)):
    """How many sides of box (x0, y0, x1, y1, on the copy) are lined with texture along MIN_SIDE_CONTACT of
    their length (left, right, top, bottom); a side with check False (a cut line or the image border) counts as seen."""
    x0, y0, x1, y1 = box
    b = SIDE_BAND
    bands = (mask[y0:y1, x0:min(x1, x0 + b)].max(axis=1), mask[y0:y1, max(x0, x1 - b):x1].max(axis=1),
             mask[y0:min(y1, y0 + b), x0:x1].max(axis=0), mask[max(y0, y1 - b):y1, x0:x1].max(axis=0))
    return sum(1 for s, c in zip(bands, check) if not c or (s.size and s.mean() >= MIN_SIDE_CONTACT))


class _Cuts:
    """Prefix sums of the cut copy, so each part's line statistics cost O(lines), not O(pixels)."""

    def __init__(self, rgb):
        # Exact integer sums: at most 400 x 255 ** 2, well inside int32.
        a = rgb.astype(np.int32)
        a2 = a * a

        def z(x, ax):  # prefix sums with a leading zero line, written in place
            shape = list(x.shape)
            shape[ax] += 1
            out = np.zeros(shape, np.int32)
            np.cumsum(x, axis=ax, dtype=np.int32, out=out[(slice(None),) * ax + (slice(1, None),)])
            return out

        # per column, sums down the rows (for column statistics) and per row, sums along the columns
        self.c1, self.c2 = z(a, 0), z(a2, 0)
        self.r1, self.r2 = z(a, 1), z(a2, 1)
        del a2
        self.hits = {}
        i16 = rgb.astype(np.int16)
        for axis in (1, 0):
            b = i16 if axis == 1 else i16.transpose(1, 0, 2)
            # Step across two positions (i and i + 2): a line that fell between two pixels of the copy
            # blends into one of them, which is dropped with the cut.
            d = np.abs(b[:, 2:] - b[:, :-2]).sum(axis=2, dtype=np.int16)  # at most 3 x 255
            m = d.shape[1]
            pad = np.pad(d, ((0, 0), (2, 2)), mode="edge")
            neigh = np.maximum.reduce([pad[:, k:k + m] for k in (0, 1, 3, 4)])
            hit = (d >= neigh) & (d >= CUT_MIN_STEP)
            # A line exactly between two pixels gives two equal steps side by side: either one counts.
            hit = (hit[:, :-1] | hit[:, 1:]).astype(np.int32)
            self.hits[axis] = np.concatenate([np.zeros((1, hit.shape[1]), np.int32), np.cumsum(hit, axis=0)], axis=0)
        H, W = rgb.shape[:2]
        flat = self.uniform((0, 0, W, H), 0)
        self.bg = None
        if flat.any():
            means = (self.r1[flat, W] / W)
            codes = (np.round(means / BAND_COLOUR_STEP)).astype(np.int64)
            keys, counts = np.unique(codes[:, 0] * 1_000_000 + codes[:, 1] * 1000 + codes[:, 2], return_counts=True)
            k = keys[np.argmax(counts)]
            self.bg = np.array([k // 1_000_000, (k // 1000) % 1000, k % 1000], np.float64) * BAND_COLOUR_STEP

    def line_means(self, box, axis):
        x0, y0, x1, y1 = box
        if axis == 1:
            return (self.c1[y1, x0:x1] - self.c1[y0, x0:x1]) / (y1 - y0)
        return (self.r1[y0:y1, x1] - self.r1[y0:y1, x0]) / (x1 - x0)

    def uniform(self, box, axis):
        """Per line of the box (columns when axis is 1, rows when 0): True when it is one flat colour."""
        x0, y0, x1, y1 = box
        if axis == 1:
            n = y1 - y0
            s1, s2 = self.c1[y1, x0:x1] - self.c1[y0, x0:x1], self.c2[y1, x0:x1] - self.c2[y0, x0:x1]
        else:
            n = x1 - x0
            s1, s2 = self.r1[y0:y1, x1] - self.r1[y0:y1, x0], self.r2[y0:y1, x1] - self.r2[y0:y1, x0]
        var = np.maximum(s2 / n - (s1 / n) ** 2, 0)
        return var.max(axis=-1) < UNIFORM_STD ** 2

    def share(self, box, axis):
        """Per position across the box: share of the line where the step is a local maximum (positions i + 1)."""
        x0, y0, x1, y1 = box
        if axis == 1:
            h = self.hits[1]
            return (h[y1, x0:max(x0, x1 - 3)] - h[y0, x0:max(x0, x1 - 3)]) / max(1, y1 - y0)
        h = self.hits[0]
        return (h[x1, y0:max(y0, y1 - 3)] - h[x0, y0:max(y0, y1 - 3)]) / max(1, x1 - x0)


def _trim(cuts, box):
    """The box without its uniform margins (page background, letterbox, gutter), or None if all uniform."""
    x0, y0, x1, y1 = box
    cols = np.nonzero(~cuts.uniform(box, 1))[0]
    rows = np.nonzero(~cuts.uniform(box, 0))[0]
    if not len(cols) or not len(rows):
        return None
    return (x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1)


def _cut(cuts, box, steps=True):
    """Where to cut the part: (axis, a, b) keeps [0, a) and [b, n) along the axis (1: a vertical line), or None.
    A uniform band across the part wins (the widest); else the straightest colour step, if strong enough."""
    x0, y0, x1, y1 = box
    best = None
    for axis in (1, 0):
        n, length = (x1 - x0, y1 - y0) if axis == 1 else (y1 - y0, x1 - x0)
        lo, hi = int(n * CUT_RANGE[0]), int(np.ceil(n * CUT_RANGE[1]))
        if hi - lo < 3 or length < 8:
            continue
        uni = cuts.uniform(box, axis)
        if uni[lo:hi].any():
            means = cuts.line_means(box, axis)
            # runs of uniform lines of one colour
            j = 0
            while j < n:
                if not uni[j]:
                    j += 1
                    continue
                k = j + 1
                while k < n and uni[k] and np.abs(means[k] - means[k - 1]).max() <= BAND_COLOUR_STEP:
                    k += 1
                if j > 0 and k < n and k > lo and j < hi:  # inside the part, not a margin
                    score = 2.0 + (k - j) / n
                    if cuts.bg is not None and np.abs(means[j:k].mean(axis=0) - cuts.bg).max() <= BAND_COLOUR_STEP:
                        score += 1.0
                    if best is None or score > best[0]:
                        best = (score, axis, j, k)
                j = k
        if not steps or hi - 3 <= lo:
            continue
        share = cuts.share(box, axis)
        i = lo + int(np.argmax(share[lo:hi - 3]))
        if share[i] >= CUT_MIN_SHARE and (best is None or share[i] > best[0]):
            best = (float(share[i]), axis, i + 1, i + 3)
    return None if best is None else best[1:]


def _leaves(cuts, box, depth, out, min_side):
    """Leaves of the recursive cuts: (trimmed box, box between the cut lines, which sides were trimmed)."""
    cut_box = box
    box = _trim(cuts, box)
    if box is None:
        return
    x0, y0, x1, y1 = box
    c = None
    if depth < MAX_CUT_DEPTH and len(out) < MAX_LEAVES and min(x1 - x0, y1 - y0) >= min_side:
        c = _cut(cuts, box, depth < MAX_STEP_DEPTH)
    if c is None:
        trimmed = (x0 - cut_box[0] > 1, cut_box[2] - x1 > 1, y0 - cut_box[1] > 1, cut_box[3] - y1 > 1)
        out.append((box, cut_box, trimmed))
        return
    axis, a, b = c
    parts = [(x0, y0, x0 + a, y1), (x0 + b, y0, x1, y1)] if axis == 1 else [(x0, y0, x1, y0 + a), (x0, y0 + b, x1, y1)]
    for p in parts:
        if p[2] > p[0] and p[3] > p[1]:
            _leaves(cuts, p, depth + 1, out, min_side)


def _natural(rgb, box):
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    if crop.shape[0] * crop.shape[1] > NATURAL_SUBSAMPLE_PIXELS:  # every other pixel is enough to count colours
        crop = crop[::2, ::2]
    crop = crop >> COLOR_BITS_DROPPED
    codes = (crop[..., 0].astype(np.int32) << 8) | (crop[..., 1].astype(np.int32) << 4) | crop[..., 2]
    colours = np.count_nonzero(np.bincount(codes.ravel(), minlength=1 << (3 * (8 - COLOR_BITS_DROPPED))))
    return min(1.0, colours / NATURAL_COLORS)


def photo_rects_part(im, ocr=None):
    """Rectangles of natural photo inside a screenshot or collage (plan phase 4: region fingerprints).
    ocr: the OCR items of the same image, when read (their boxes are text, never photo).
    Returns (rects, flat_fraction, grid): grid is the number of photo cells when the image is tiled by
    photos (a collage), else 0."""
    W, H = im.size
    scale = min(1.0, RECT_MAX_SIDE / max(W, H))
    if scale < 1:
        # Pillow's integer box reduction first (no full-size copy of a 4096 px original: about 50 MB),
        # then OpenCV's area filter to the exact size (the same as a box filter, several times faster).
        factor = int(1 / scale)
        small = im.reduce(factor) if factor >= 2 else im
        rgb = cv2.resize(np.asarray(small), (max(1, round(W * scale)), max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
        del small
    else:
        rgb = np.asarray(im)
    sh, sw = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (FLAT_WINDOW, FLAT_WINDOW))
    var = cv2.boxFilter(gray * gray, -1, (FLAT_WINDOW, FLAT_WINDOW)) - mean * mean
    textured = (np.sqrt(np.maximum(var, 0)) >= FLAT_STD).astype(np.uint8)
    flat_fraction = float(1.0 - textured.mean())
    for o in ocr or []:
        bx = o.get("box") if isinstance(o, dict) else o
        x0, y0, x1, y1 = (int(round(v * scale)) for v in bx)
        textured[max(0, y0 - OCR_PAD):min(sh, y1 + OCR_PAD), max(0, x0 - OCR_PAD):min(sw, x1 + OCR_PAD)] = 0
    side = min(sw, sh)
    k_close = max(3, int(side * CLOSE_FRACTION) | 1)
    k_open = max(3, int(side * OPEN_FRACTION) | 1)
    mask = cv2.morphologyEx(textured, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))

    candidates = []  # (box, which sides need the side test, fallback box)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not cw * ch:
            continue
        fill = cv2.contourArea(c) / (cw * ch)
        if fill >= MIN_RECT_FILL:
            candidates.append(((x, y, x + cw, y + ch), None, None))
        elif fill >= MIN_BLOB_FILL:
            candidates.append(((x, y, x + cw, y + ch), (True, True, True, True), None))
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_RECT_FRACTION * sw * sh
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_area:  # its polygon could not be large enough
            continue
        poly = cv2.approxPolyDP(c, POLY_EPSILON * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        x, y, cw, ch = cv2.boundingRect(poly)
        if cw * ch and cv2.contourArea(poly) / (cw * ch) >= MIN_RECT_FILL:
            candidates.append(((x, y, x + cw, y + ch), None, None))
    ck = min(1.0, CUT_MAX_SIDE / max(sw, sh))
    tiny = rgb if ck >= 1 else cv2.resize(rgb, (max(1, round(sw * ck)), max(1, round(sh * ck))), interpolation=cv2.INTER_AREA)
    tleaves = []
    if min(tiny.shape[:2]) >= MIN_CUT_SIDE:  # a long thin image (130:1 and more) has nothing to cut
        _leaves(_Cuts(tiny), (0, 0, tiny.shape[1], tiny.shape[0]), 0, tleaves, int(round(min(tiny.shape[:2]) * MIN_LEAF_SIDE)))
    up = lambda b: (int(b[0] / ck), int(b[1] / ck), min(sw, int(np.ceil(b[2] / ck))), min(sh, int(np.ceil(b[3] / ck))))
    leaves = []
    for b, cb, trimmed in tleaves:
        b, cb = up(b), up(cb)
        leaves.append(cb)
        # The trimmed leaf when its trimmed sides look like a photo's edges; else the box between the cut
        # lines (a photo whose own black or white border was trimmed away, a collage cell).
        candidates.append((b, trimmed, cb if any(trimmed) and cb != (0, 0, sw, sh) else None))

    def fits(box):
        cw, ch = box[2] - box[0], box[3] - box[1]
        frac = cw * ch / float(sw * sh)
        return MIN_RECT_FRACTION <= frac <= MAX_RECT_FRACTION and max(cw, ch) / max(1, min(cw, ch)) <= MAX_RECT_ASPECT, frac

    scored, seen = [], set()
    for box, check, fallback in candidates:
        ok, frac = fits(box)
        natural = _natural(rgb, box) if ok else 0.0
        if ok and natural >= 0.5 and check is not None and _sides_seen(textured, box, check) < MIN_SIDES_SEEN:
            ok = False
            if fallback is not None:
                box = fallback
                ok, frac = fits(box)
                natural = _natural(rgb, box) if ok else 0.0
        if not ok or natural < 0.5 or tuple(box) in seen:
            continue
        seen.add(tuple(box))
        scored.append((frac * natural, list(box), natural, frac))
    scored.sort(key=lambda s: -s[0])
    kept = []
    for score, box, natural, frac in scored:
        if all(_iou(box, k["_small"]) < RECT_IOU_SAME for k in kept):
            kept.append({"_small": box, "score": round(score, 4), "natural": round(natural, 3), "_frac": frac})
        if len(kept) >= MAX_PHOTO_RECTS:
            break
    # A collage: photo cells that tile the image (leaves of the cuts, each big enough).
    cells = [b for b in leaves if (b[2] - b[0]) * (b[3] - b[1]) >= GRID_MIN_CELL_FRACTION * sw * sh and _natural(rgb, b) >= 0.5]
    cover = sum((b[2] - b[0]) * (b[3] - b[1]) for b in cells) / float(sw * sh)
    grid = len(cells) if len(cells) >= GRID_MIN_CELLS and cover >= GRID_MIN_COVER else 0
    W, H = im.size
    rects = []
    for k in kept:
        b = k.pop("_small")
        k.pop("_frac")
        k["box"] = [int(b[0] / scale), int(b[1] / scale), min(W, int(np.ceil(b[2] / scale))), min(H, int(np.ceil(b[3] / scale)))]
        rects.append(k)
    return rects, flat_fraction, grid


def screenshot_signs(im, ocr, rects, flat_fraction, grid=None):
    w, h = im.size
    ratio = max(w, h) / min(w, h)
    signs = {
        "phone_aspect": bool(h > w and PHONE_ASPECT_RANGE[0] <= ratio <= PHONE_ASPECT_RANGE[1]),
        "screen_size": (w, h) in SCREEN_SIZES or (h, w) in SCREEN_SIZES,
        "flat_fraction": None if flat_fraction is None else round(flat_fraction, 4),
        "photo_rect_fraction": None,
        "collage_cells": grid,
        "status_bar": None, "status_bar_text": None, "text_fraction": None, "text_lines": None,
    }
    if rects is not None:
        signs["photo_rect_fraction"] = round(max([(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) / (w * h) for r in rects], default=0.0), 4)
    if ocr is not None:
        top = [o["text"] for o in ocr if o["box"][3] <= h * STATUS_BAR_FRACTION]
        hit = next((t for t in top if TIME_RE.search(t) or BATTERY_RE.search(t)), None)
        signs["status_bar"] = hit is not None
        signs["status_bar_text"] = hit
        signs["text_lines"] = len(ocr)
        signs["text_fraction"] = round(min(1.0, sum((o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) for o in ocr) / (w * h)), 4)
    return signs


def screen(reg, im, meta):
    parts = meta.get("parts", list(DEFAULT_PARTS))
    if not isinstance(parts, list) or not parts or any(p not in PARTS for p in parts):
        raise BadInput(400, "bad_meta", f"parts must be a non-empty list taken from {list(PARTS)}")
    grid = meta.get("tiles", DEFAULT_TILE_GRID)
    if not isinstance(grid, int) or isinstance(grid, bool) or not 0 <= grid <= MAX_TILE_GRID:
        raise BadInput(400, "bad_meta", f"tiles must be an integer from 0 to {MAX_TILE_GRID}")
    w, h = im.size
    out, ms = {"size": [w, h]}, {}

    def timed(name, fn):
        t = time.perf_counter()
        r = fn()
        ms[name] = round((time.perf_counter() - t) * 1000)
        return r

    want_hot = "hotspots" in parts and grid > 0
    if "siglip" in parts or want_hot:
        out.update(timed("siglip", lambda: siglip_part(reg, im, grid, want_hot)))
    ocr = timed("ocr", lambda: ocr_part(reg, im)) if "ocr" in parts else None
    if ocr is not None:
        out["ocr"] = ocr
    if "faces" in parts:
        out["faces"] = timed("faces", lambda: faces_part(reg, im))
    rects, flat, cells = (None, None, None)
    if "photo_rects" in parts:
        rects, flat, cells = timed("photo_rects", lambda: photo_rects_part(im, ocr))
        out["photo_rects"] = rects
    out["screenshot_signs"] = screenshot_signs(im, ocr, rects, flat, cells)
    out["timings_ms"] = ms
    return out
