#!/usr/bin/env python3
"""Calibration of the reused-photo check on real photos (design note 40, phase 4b).

Set: 4,000 distinct Open Images V7 validation photos (fetch-openimages.py, openimages-manifest.json),
2,000 « calibrate » and 2,000 « measure ». The copies (9 types of oi_transforms.py) are made in memory,
seeded, never written. Every threshold is chosen on the calibrate half; the measure half only measures.
The vision service's own modules are imported (no HTTP); `/verify` is replayed with the same OpenCV
calls and SIFT features cached per image (checked equal to verify.verify on a sample).

Measures: the cascade of services/photoMatch.ts (PDQ <= 31 with quality > 49, DINOv2 >= T, /verify >= K),
then each improvement: region fingerprints of the new rectangle finder (service/app/screen.py), fixed
query tiles (2 x 2 and centre), the near-identity rule (PDQ <= 4 and cosine >= 0.99 without /verify),
extra stored DINOv2 vectors (70 % crops). Then the ABO set of phase 1 again with the new values
(« catalogue stress test », from its cache). It never writes config/photo-checks.json.

Run (from the repository root), with the venv of calibrate-fingerprints.py:
  MODELS_DIR=artifacts/vision-models nice -n 10 <venv>/bin/python -I eval/calibrate-openimages.py \
      [all|originals|queries|calibrate|measure|abo|report]
Cache: <artifacts>/eval-photos/openimages-cache/ (a stage recomputes only what is missing).
"""
import os

WORKERS = int(os.environ.get("FP_WORKERS", "4"))
os.environ.setdefault("ORT_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(WORKERS))
os.environ.setdefault("DINO_VARIANT", "small")

import base64
import collections
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
# The vision service's own modules (service/app), imported directly.
sys.path.insert(0, os.path.join(REPO, "service"))
sys.path.insert(0, HERE)
ART = os.environ.get("EVAL_PHOTOS_DIR", os.path.join(REPO, "artifacts", "eval-photos"))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO, "artifacts", "vision-models"))
OI_DIR = os.path.join(ART, "openimages")
# OI_LIMIT: a smoke test on the first N photos of each half, in its own cache.
OI_LIMIT = int(os.environ.get("OI_LIMIT", "0"))
CACHE = os.path.join(ART, "openimages-cache" + ("-smoke" if OI_LIMIT else ""))
MANIFEST = os.path.join(HERE, "openimages-manifest.json")
# The cascade's starting values (PDQ distance and quality, minimum inliers), as in the API's configuration
# when the run was made; only these three are read. PHOTO_CHECKS_CONFIG points to another file.
CONFIG = os.environ.get("PHOTO_CHECKS_CONFIG", os.path.join(HERE, "fingerprint-config.json"))
REPORT = os.path.join(ART, "calibration-fingerprints-2026-11-25.md")
ABO_FP = os.path.join(ART, "fingerprint")
ABO_CACHE = os.path.join(ABO_FP, "cache")

import oi_transforms as TR  # noqa: E402
from app.fingerprint import tile_boxes  # noqa: E402  the fixed query tiles of /fingerprint ("tiles": true)

SEED = 20261125
# Sizes chosen for the laptop's time (about 4 h with 4 processes on a shared machine):
CAL_PARENTS = 600           # calibrate parents whose copies are made
MEASURE_PARENTS = 1000      # measure parents whose copies are made (the database: all 2,000 measure originals)
TILE_SAMPLE = 150           # measure copies of the non-region types that also get tiles (per type)
ORIG_TILES = 500            # originals per half fingerprinted with tiles and rectangles (ordinary uploads)
VERIFY_POS = 2000           # /verify positives (calibrate), stratified by type
VERIFY_NEG = 2000           # /verify hard negatives (calibrate): 1,000 original pairs, 1,000 copy-vs-wrong
REGION_NEG = 500            # extra hard negatives: regions against their most similar wrong original (box_a)
MIN_FREE_GB = 4.0
MAX_VERIFY_PER_STAGE = 4    # services/photoMatch.ts
PDQ_BITS = 256
NEAR_PDQ = 4                # near-identity rule (brief): PDQ <= 4 with both qualities > minQuality...
NEAR_COS = 0.99             # ... and DINOv2 cosine >= 0.99
TARGET_FMR = 1e-7
SOTA_SCREENSHOT = 0.18
STORED_CROPS = {"c70": (0.15, 0.15, 0.85, 0.85), "tl70": (0.0, 0.0, 0.7, 0.7), "br70": (0.3, 0.3, 1.0, 1.0)}
EXTRA_SETS = {"+c70": ["c70"], "+tl70+br70": ["tl70", "br70"]}
KINDS = {"new": 1, "old": 2, "oracle": 3, "tile": 4, "both": 5}  # both: found by the new and the old finder
TYPES = TR.TYPES
REGION_TYPES = TR.REGION_TYPES


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check_disk():
    free = shutil.disk_usage(ART).free / 1e9
    if free < MIN_FREE_GB:
        sys.exit(f"stop: {free:.2f} GB free on the disk, under {MIN_FREE_GB} GB")
    return free


def load_config():
    with open(CONFIG) as f:
        c = json.load(f)
    return {"maxDistance": c["pdq"]["maxDistance"]["value"], "minQuality": c["pdq"]["minQuality"]["value"],
            "minInliers": c["verify"]["minInliers"]["value"]}


def manifest():
    with open(MANIFEST) as f:
        m = json.load(f)["images"]
    if OI_LIMIT:
        cnt = collections.Counter()
        keep = []
        for x in m:
            cnt[x["half"]] += 1
            if cnt[x["half"]] <= OI_LIMIT:
                keep.append(x)
        m = keep
    return m


def seeded(n, k, tag):
    rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{tag}".encode()).hexdigest()[:16], 16))
    return np.sort(rng.choice(n, min(k, n), replace=False))


def crop_box(w, h, f):
    return [round(w * f[0]), round(h * f[1]), round(w * f[2]), round(h * f[3])]


# ---- the committed rectangle finder (9da0efc), for the « before » column --------------------------

def old_photo_rects(im):
    import cv2
    from app.images import shrink
    from app import screen as S
    small, scale = shrink(im, S.RECT_MAX_SIDE)
    rgb = np.asarray(small)
    sh, sw = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (S.FLAT_WINDOW, S.FLAT_WINDOW))
    var = cv2.boxFilter(gray * gray, -1, (S.FLAT_WINDOW, S.FLAT_WINDOW)) - mean * mean
    textured = (np.sqrt(np.maximum(var, 0)) >= S.FLAT_STD).astype(np.uint8)
    side = min(sw, sh)
    k_close = max(3, int(side * S.CLOSE_FRACTION) | 1)
    k_open = max(3, int(side * S.OPEN_FRACTION) | 1)
    mask = cv2.morphologyEx(textured, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    candidates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch and cv2.contourArea(c) / (cw * ch) >= S.MIN_RECT_FILL:
            candidates.append((x, y, x + cw, y + ch))
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        poly = cv2.approxPolyDP(c, S.POLY_EPSILON * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        x, y, cw, ch = cv2.boundingRect(poly)
        if cw * ch and cv2.contourArea(poly) / (cw * ch) >= S.MIN_RECT_FILL:
            candidates.append((x, y, x + cw, y + ch))
    scored = []
    for (x0, y0, x1, y1) in candidates:
        cw, ch = x1 - x0, y1 - y0
        frac = cw * ch / float(sw * sh)
        if not (S.MIN_RECT_FRACTION <= frac <= S.MAX_RECT_FRACTION) or max(cw, ch) / max(1, min(cw, ch)) > S.MAX_RECT_ASPECT:
            continue
        crop = rgb[y0:y1, x0:x1] >> S.COLOR_BITS_DROPPED
        codes = (crop[..., 0].astype(np.int32) << 8) | (crop[..., 1].astype(np.int32) << 4) | crop[..., 2]
        natural = min(1.0, len(np.unique(codes)) / S.NATURAL_COLORS)
        if natural < 0.5:
            continue
        scored.append((frac * natural, [x0, y0, x1, y1]))
    scored.sort(key=lambda s: -s[0])
    kept = []
    for score, box in scored:
        if all(S._iou(box, k) < S.RECT_IOU_SAME for k in kept):
            kept.append(box)
        if len(kept) >= S.MAX_PHOTO_RECTS:
            break
    W, H = im.size
    return [[int(b[0] / scale), int(b[1] / scale), min(W, int(np.ceil(b[2] / scale))), min(H, int(np.ceil(b[3] / scale)))] for b in kept]


# ---- workers -------------------------------------------------------------------------------------
_REG = None
_IDS = None
_IMG_CACHE = collections.OrderedDict()


def _init_worker():
    global _REG, _IDS
    import cv2
    cv2.setNumThreads(1)
    from app import models
    _REG = models.Registry()
    _REG.dino()
    _IDS = [x["id"] for x in manifest()]


def _path(i):
    return os.path.join(OI_DIR, i + ".jpg")


def _load(i, sha=None):
    """Upright RGB image of an Open Images photo (small LRU: fillers come back often)."""
    from app import images
    if i in _IMG_CACHE:
        _IMG_CACHE.move_to_end(i)
        return _IMG_CACHE[i]
    with open(_path(i), "rb") as f:
        data = f.read()
    if sha is not None and hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"{i}: sha256 does not match the manifest")
    im = images.decode(data)
    _IMG_CACHE[i] = im
    if len(_IMG_CACHE) > 48:
        _IMG_CACHE.popitem(last=False)
    return im


def _pool_of(half):
    return [x["id"] for x in manifest() if x["half"] == half]


_POOLS = {}


def make_query(half, kind, parent):
    """(bytes, info) of a copy, regenerated deterministically."""
    if half not in _POOLS:
        _POOLS[half] = _pool_of(half)
    return TR.make(parent, kind, _load(parent), _POOLS[half], _load)


def _hex8(hexes):
    return np.stack([np.frombuffer(bytes.fromhex(h), np.uint8) for h in hexes])


def whole_fp(im):
    from app import fingerprint
    out = fingerprint.fingerprint(_REG, im, {})
    return _hex8(out["pdq"]), int(out["quality"]), np.frombuffer(base64.b64decode(out["dino"]), "<f2").copy()


def region_fp(im, boxes, with_pdq=True):
    """What /fingerprint returns for its regions (same preprocessing, same model), without the whole image."""
    from app import fingerprint, images, models
    if not boxes:
        return []
    boxes = [images.parse_box(b, *im.size) for b in boxes]
    x = np.stack([images.to_tensor(im, models.DINO_SIZE, models.DINO_RESAMPLE, models.DINO_MEAN, models.DINO_STD, b) for b in boxes])
    v = models.l2(_REG.dino().run(["last_hidden_state"], {"pixel_values": x})[0][:, 0, :].astype(np.float32))
    out = []
    for b, vec in zip(boxes, v):
        if with_pdq:
            h, q = fingerprint.pdq(im.crop(tuple(b)))
            out.append((b, _hex8(h), int(q), vec.astype("<f2")))
        else:
            out.append((b, None, None, vec.astype("<f2")))
    return out


def ok_box(b):
    from app.fingerprint import MIN_REGION_SIDE
    return min(b[2] - b[0], b[3] - b[1]) >= MIN_REGION_SIDE


def _orig_worker(item):
    from app import screen
    idx, i, sha, do_tiles = item
    t0 = time.perf_counter()
    im = _load(i, sha)
    w, h = im.size
    t1 = time.perf_counter()
    p8, q, d = whole_fp(im)
    t2 = time.perf_counter()
    extra = region_fp(im, [crop_box(w, h, f) for f in STORED_CROPS.values()], with_pdq=False)
    t3 = time.perf_counter()
    rects, flat, cells = screen.photo_rects_part(im)
    t4 = time.perf_counter()
    old = old_photo_rects(im)
    t5 = time.perf_counter()
    regs, ms_tiles, ms_rfp = [], 0.0, 0.0
    if do_tiles:
        regs += [("tile", r) for r in region_fp(im, tile_boxes(w, h))]
        t6 = time.perf_counter()
        regs += [("new", r) for r in region_fp(im, [r["box"] for r in rects if ok_box(r["box"])])]
        ms_tiles, ms_rfp = (t6 - t5) * 1000, (time.perf_counter() - t6) * 1000
    return {"idx": idx, "p8": p8, "q": q, "d": d, "extra": np.stack([e[3] for e in extra]), "size": (w, h),
            "n_new": len(rects), "n_old": len(old), "cells": cells, "flat": flat, "regs": regs,
            "ms": {"decode": (t1 - t0) * 1000, "fp": (t2 - t1) * 1000, "extra": (t3 - t2) * 1000, "rects_new": (t4 - t3) * 1000,
                   "rects_old": (t5 - t4) * 1000, "tiles": ms_tiles, "rect_fp": ms_rfp}}


def _query_worker(item):
    from app import images, screen
    half, kind, pidx, parent, do_tiles = item
    t0 = time.perf_counter()
    data, info = make_query(half, kind, parent)
    t1 = time.perf_counter()
    im = images.decode(data)
    w, h = im.size
    t2 = time.perf_counter()
    p8, q, d = whole_fp(im)
    t3 = time.perf_counter()
    rects, flat, cells = screen.photo_rects_part(im)
    t4 = time.perf_counter()
    old = old_photo_rects(im)
    t5 = time.perf_counter()
    new_boxes = [r["box"] for r in rects if ok_box(r["box"])]
    old_boxes = [b for b in old if ok_box(b)]
    regs = []
    ms_tiles = ms_new = 0.0
    if kind in REGION_TYPES or do_tiles:
        a = time.perf_counter()
        regs += [("both" if b in old_boxes else "new", r) for b, r in zip(new_boxes, region_fp(im, new_boxes))]
        ms_new = (time.perf_counter() - a) * 1000
        regs += [("old", r) for r in region_fp(im, [b for b in old_boxes if b not in new_boxes])]
        if kind in REGION_TYPES:
            regs += [("oracle", r) for r in region_fp(im, [info["box"]])]
    if do_tiles:
        a = time.perf_counter()
        regs += [("tile", r) for r in region_fp(im, tile_boxes(w, h))]
        ms_tiles = (time.perf_counter() - a) * 1000
    return {"pidx": pidx, "kind": kind, "p8": p8, "q": q, "d": d, "size": (w, h), "info": info, "regs": regs,
            "n_new": len(rects), "n_old": len(old), "cells": cells, "flat": flat, "new_boxes": new_boxes,
            "ms": {"make": (t1 - t0) * 1000, "decode": (t2 - t1) * 1000, "fp": (t3 - t2) * 1000, "rects_new": (t4 - t3) * 1000,
                   "rects_old": (t5 - t4) * 1000, "tiles": ms_tiles, "rect_fp": ms_new}}


# /verify, replayed: verify.verify's own steps, SIFT features cached per (image, box, mirrored).
_FEAT = collections.OrderedDict()


def _gray(im, box):
    from app import verify
    return verify._prep(im, box)


def _feat(key, gray_fn):
    import cv2
    from app import verify
    if key in _FEAT:
        _FEAT.move_to_end(key)
        return _FEAT[key]
    _FEAT[key] = verify._features(cv2.SIFT_create(nfeatures=verify.MAX_FEATURES), gray_fn())
    if len(_FEAT) > 64:
        _FEAT.popitem(last=False)
    return _FEAT[key]


def verify_replay(qkey, qim_fn, box, ckey, cim_fn):
    """verify.verify's inliers, from its own functions, with the SIFT features of each image cached."""
    from app import verify
    bkey = None if box is None else tuple(box)
    fa = _feat(("q", qkey, bkey), lambda: _gray(qim_fn(), box))
    fb = _feat(("c", ckey, False), lambda: _gray(cim_fn(), None))
    n = verify._match_features(fa, fb)["inliers"]
    if n < verify.TRY_MIRROR_BELOW:
        fbm = _feat(("c", ckey, True), lambda: np.ascontiguousarray(_gray(cim_fn(), None)[:, ::-1]))
        n = max(n, verify._match_features(fa, fbm)["inliers"])
    return n


def _qimage(spec):
    """spec: ('oi', idx) an Open Images original; ('copy', half, kind, parent); ('abo', path)."""
    from app import images
    if spec[0] == "oi":
        return _load(_IDS[spec[1]])
    if spec[0] == "copy":
        return images.decode(make_query(spec[1], spec[2], spec[3])[0])
    with open(spec[1], "rb") as f:
        return images.decode(f.read())


def _verify_worker(item):
    """item: (candidate spec, [(query spec, box), ...]) -> [(inliers, ms)]."""
    cspec, jobs = item
    cim = {}
    cget = lambda: cim.setdefault("im", _qimage(cspec))
    out = []
    for qspec, box in jobs:
        t = time.perf_counter()
        qim = {}
        qget = lambda: qim.setdefault("im", _qimage(qspec))
        n = verify_replay(qspec, qget, box, cspec, cget)
        out.append((n, (time.perf_counter() - t) * 1000))
    return out


def _verify_check_worker(item):
    """verify.verify itself (production code and time), against the replay."""
    from app import verify
    cspec, qspec, box = item
    a, b = _qimage(qspec), _qimage(cspec)
    t = time.perf_counter()
    r = verify.verify(a, b, {} if box is None else {"box_a": box})
    ms = (time.perf_counter() - t) * 1000
    _FEAT.clear()
    n = verify_replay(qspec, lambda: a, box, cspec, lambda: b)
    _FEAT.clear()
    return r["inliers"], n, ms


def pool_map(fn, items, label, chunk=4):
    out = []
    t = time.time()
    with mp.get_context("fork").Pool(WORKERS, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(fn, items, chunksize=chunk)):
            out.append(r)
            if (i + 1) % 500 == 0:
                log(f"  {label}: {i + 1}/{len(items)} ({(time.time() - t) / (i + 1) * 1000:.0f} ms per item, wall)")
                check_disk()
    return out


# ---- stage 1: originals ------------------------------------------------------------------------

def stage_originals():
    dest = os.path.join(CACHE, "originals.npz")
    if os.path.exists(dest):
        return
    os.makedirs(CACHE, exist_ok=True)
    check_disk()
    imgs = manifest()
    tiles_for = set()
    for half in ("calibrate", "measure"):
        idx = [k for k, x in enumerate(imgs) if x["half"] == half]
        tiles_for |= {idx[j] for j in seeded(len(idx), ORIG_TILES, f"orig-tiles-{half}")}
    log(f"originals: {len(imgs)} photos ({len(tiles_for)} with tiles and rectangles)")
    res = pool_map(_orig_worker, [(k, x["id"], x["sha256"], k in tiles_for) for k, x in enumerate(imgs)], "originals")
    res.sort(key=lambda r: r["idx"])
    save_regions = []
    for r in res:
        for kind, (b, p8, q, d) in r["regs"]:
            save_regions.append((r["idx"], KINDS[kind], b, p8, q, d))
    np.savez(dest, ids=np.array([x["id"] for x in imgs]), half=np.array([x["half"] for x in imgs]),
             pdq=np.stack([r["p8"] for r in res]), quality=np.array([r["q"] for r in res], np.int16),
             dino=np.stack([r["d"] for r in res]).astype("<f2"), extra=np.stack([r["extra"] for r in res]).astype("<f2"),
             size=np.array([r["size"] for r in res], np.int32), n_new=np.array([r["n_new"] for r in res], np.int16),
             n_old=np.array([r["n_old"] for r in res], np.int16), cells=np.array([r["cells"] for r in res], np.int16),
             flat=np.array([r["flat"] for r in res], np.float32), tiled=np.array([r["idx"] in tiles_for for r in res]),
             r_owner=np.array([s[0] for s in save_regions], np.int32), r_kind=np.array([s[1] for s in save_regions], np.int8),
             r_box=np.array([s[2] for s in save_regions], np.int32).reshape(-1, 4),
             r_pdq=np.array([s[3] for s in save_regions], np.uint8).reshape(-1, 8, 32),
             r_q=np.array([s[4] for s in save_regions], np.int16), r_dino=np.array([s[5] for s in save_regions], "<f2").reshape(-1, 384),
             **{f"ms_{k}": np.array([r["ms"][k] for r in res], np.float32) for k in res[0]["ms"]})
    log(f"  saved {dest}")


def load_npz(name):
    z = np.load(os.path.join(CACHE, name), allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---- stage 2: the copies -----------------------------------------------------------------------

def query_plan(half):
    imgs = manifest()
    idx = [k for k, x in enumerate(imgs) if x["half"] == half]
    parents = [idx[j] for j in seeded(len(idx), MEASURE_PARENTS if half == "measure" else CAL_PARENTS, f"{half}-parents")]
    items = []
    for kind in TYPES:
        if kind in REGION_TYPES:
            tiled = set(parents)
        else:
            tiled = {parents[j] for j in seeded(len(parents), TILE_SAMPLE, f"tiles-{half}-{kind}")}
        items += [(half, kind, p, imgs[p]["id"], p in tiled) for p in parents]
    return items


def stage_queries():
    for half in ("calibrate", "measure"):
        for kind in TYPES:
            dest = os.path.join(CACHE, f"q-{half}-{kind}.npz")
            if os.path.exists(dest):
                continue
            check_disk()
            items = [x for x in query_plan(half) if x[1] == kind]
            log(f"queries {half} {kind}: {len(items)} copies ({sum(x[4] for x in items)} with tiles)")
            res = pool_map(_query_worker, items, f"{half} {kind}")
            regs = []
            for k, r in enumerate(res):
                for rk, (b, p8, q, d) in r["regs"]:
                    regs.append((k, KINDS[rk], b, p8, q, d))
            np.savez(dest, pidx=np.array([r["pidx"] for r in res], np.int32), pdq=np.stack([r["p8"] for r in res]),
                     quality=np.array([r["q"] for r in res], np.int16), dino=np.stack([r["d"] for r in res]).astype("<f2"),
                     size=np.array([r["size"] for r in res], np.int32), n_new=np.array([r["n_new"] for r in res], np.int16),
                     n_old=np.array([r["n_old"] for r in res], np.int16), cells=np.array([r["cells"] for r in res], np.int16),
                     flat=np.array([r["flat"] for r in res], np.float32), tiled=np.array([x[4] for x in items]),
                     info=np.array([json.dumps(r["info"]) for r in res]),
                     r_owner=np.array([s[0] for s in regs], np.int32), r_kind=np.array([s[1] for s in regs], np.int8),
                     r_box=np.array([s[2] for s in regs], np.int32).reshape(-1, 4),
                     r_pdq=np.array([s[3] for s in regs], np.uint8).reshape(-1, 8, 32),
                     r_q=np.array([s[4] for s in regs], np.int16), r_dino=np.array([s[5] for s in regs], "<f2").reshape(-1, 384),
                     **{f"ms_{k}": np.array([r["ms"][k] for r in res], np.float32) for k in res[0]["ms"]})
            log(f"  saved {dest}")


# ---- maths ---------------------------------------------------------------------------------------

def as_u64(pdq):
    return np.ascontiguousarray(pdq).view(">u8").astype(np.uint64)


def f32(d):
    return np.asarray(d, dtype="<f2").astype(np.float32)


def pdq_min_dist(q8, db_up):
    """(n, 8, 4) query orientations against (m, 4) stored upright hashes: (n, m) best distance."""
    best = np.full((q8.shape[0], db_up.shape[0]), PDQ_BITS, np.uint16)
    for o in range(8):
        d = np.zeros_like(best)
        for w in range(4):
            d += np.bitwise_count(q8[:, o, w][:, None] ^ db_up[None, :, w]).astype(np.uint16)
        np.minimum(best, d, out=best)
    return best


def pwm_gpd(excess):
    y = np.sort(excess)
    n = len(y)
    a0 = y.mean()
    a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * y) / n
    k = a0 / (a0 - 2 * a1) - 2
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    return float(k), float(sigma)


def gpd_quantile(u, k, sigma, ratio):
    return u + (sigma / k * (1 - ratio ** k) if abs(k) > 1e-9 else -sigma * math.log(ratio))


def levels(top, pairs, fmrs=(1e-4, 1e-5, 1e-6, 1e-7)):
    """Empirical thresholds (cosines strictly above are at most floor(f * pairs)) and a GPD tail model."""
    top = np.sort(np.asarray(top))[::-1]
    emp = {}
    for f in fmrs:
        k = int(math.floor(f * pairs))
        emp[f] = float(top[k]) if k < len(top) else None
    u = emp[1e-4]
    n_u = int(np.sum(top > u)) if u is not None else 0
    if n_u >= 50:
        kk, sig = pwm_gpd(top[:n_u] - u)
        model = {f: gpd_quantile(u, kk, sig, f * pairs / n_u) for f in (1e-5, 1e-6, 1e-7, 1e-8)}
        gpd = {"u": u, "n_u": n_u, "k": kk, "sigma": sig, "q": model, "endpoint": (u + sig / kk) if kk > 0 else None}
    else:  # too few pairs above the 1e-4 level for a tail model
        gpd = {"u": u, "n_u": n_u, "q": {}, "endpoint": None}
    return {"emp": emp, "gpd": gpd, "max": float(top[0]), "tail": [float(x) for x in top[:15]], "pairs": int(pairs)}


# ---- half data -----------------------------------------------------------------------------------

class Half:
    """The originals of one half (the database) and their copies (the queries)."""

    def __init__(self, half):
        O = load_npz("originals.npz")
        self.half = half
        self.gidx = np.nonzero(O["half"] == half)[0]
        self.ids = O["ids"][self.gidx]
        self.pos = {int(g): k for k, g in enumerate(self.gidx)}
        self.D = f32(O["dino"][self.gidx])
        self.E = {c: f32(O["extra"][self.gidx, k]) for k, c in enumerate(STORED_CROPS)}
        self.Q8 = as_u64(O["pdq"][self.gidx])
        self.up = self.Q8[:, 0, :]
        self.PQ = O["quality"][self.gidx]
        self.O = O
        sel = np.isin(O["r_owner"], self.gidx)
        self.oreg = {"owner": np.array([self.pos[int(g)] for g in O["r_owner"][sel]], np.int64), "kind": O["r_kind"][sel],
                     "box": O["r_box"][sel], "d": f32(O["r_dino"][sel]), "p8": as_u64(O["r_pdq"][sel]), "q": O["r_q"][sel]}
        self.tiled = O["tiled"][self.gidx]
        self.queries = {}
        for kind in TYPES:
            p = os.path.join(CACHE, f"q-{half}-{kind}.npz")
            if not os.path.exists(p):
                continue
            z = load_npz(f"q-{half}-{kind}.npz")
            n = len(z["pidx"])
            info = [json.loads(s) for s in z["info"]]
            fill = [{self.idx_of(f) for f in x.get("fillers", [])} for x in info]
            self.queries[kind] = {"parent": np.array([self.pos[int(g)] for g in z["pidx"]]), "pidx": z["pidx"],
                                  "d": f32(z["dino"]), "p8": as_u64(z["pdq"]), "q": z["quality"], "info": info, "fill": fill,
                                  "tiled": z["tiled"], "cells": z["cells"], "flat": z["flat"], "n_new": z["n_new"], "n_old": z["n_old"],
                                  "reg": {"owner": z["r_owner"].astype(np.int64), "kind": z["r_kind"], "box": z["r_box"],
                                          "d": f32(z["r_dino"]), "p8": as_u64(z["r_pdq"]), "q": z["r_q"]},
                                  "ms": {k[3:]: z[k] for k in z if k.startswith("ms_")}, "n": n,
                                  "signs": (z["n_new"] > 0) | (z["cells"] >= 2)}
        # the unchanged re-upload: the stored file itself
        self.queries = {"reupload": {"parent": np.arange(len(self.gidx)), "pidx": self.gidx, "d": self.D, "p8": self.Q8, "q": self.PQ,
                                     "info": [{} for _ in self.gidx], "fill": [set() for _ in self.gidx],
                                     "tiled": np.zeros(len(self.gidx), bool), "n": len(self.gidx),
                                     "signs": (O["n_new"][self.gidx] > 0) | (O["cells"][self.gidx] >= 2),
                                     "reg": {"owner": np.zeros(0, np.int64), "kind": np.zeros(0, np.int8), "box": np.zeros((0, 4), np.int32),
                                             "d": np.zeros((0, 384), np.float32), "p8": np.zeros((0, 8, 4), np.uint64), "q": np.zeros(0, np.int16)}},
                        **self.queries}

    def idx_of(self, i):
        if not hasattr(self, "_gpos"):
            self._gpos = {str(x): g for g, x in enumerate(self.O["ids"])}
        return self.pos.get(self._gpos[i], -1)

    def qspec(self, kind, r):
        Q = self.queries[kind]
        if kind == "reupload":
            return ("oi", int(Q["pidx"][r]))
        return ("copy", self.half, kind, str(self.O["ids"][int(Q["pidx"][r])]))

    def cspec(self, j):
        return ("oi", int(self.gidx[j]))


def pair_scan(H, keep_from, cfg, extras=None):
    """All unordered pairs of distinct originals of a half: top cosines (>= keep_from), PDQ pairs, the
    near-identity pairs. With extras (a list of crop names): the pair score is the best of the query's
    whole vector against the other photo's whole and extra vectors, in both directions."""
    D, n = H.D, len(H.D)
    top_i, top_j, top_c, pdq_pairs = [], [], [], []
    near = []
    step = 400
    for s in range(0, n, step):
        e = min(n, s + step)
        S = D[s:e] @ D.T
        if extras:
            for c in extras:
                S = np.maximum(S, D[s:e] @ H.E[c].T)          # row photo as query, column photo's crops stored
                S = np.maximum(S, (D @ H.E[c][s:e].T).T)        # column photo as query, row photo's crops stored
        iu = np.arange(s, e)[:, None] < np.arange(n)[None, :]
        ii, jj = np.nonzero(iu & (S >= keep_from))
        top_i.append(ii + s); top_j.append(jj); top_c.append(S[ii, jj])
        if extras is None:
            d1 = pdq_min_dist(H.Q8[s:e], H.up)
            d2 = pdq_min_dist(H.Q8, H.up[s:e]).T
            dmin = np.minimum(d1, d2)
            okq = (H.PQ[s:e, None] > cfg["minQuality"]) & (H.PQ[None, :] > cfg["minQuality"])
            for a, b in zip(*np.nonzero(iu & (dmin <= cfg["maxDistance"]) & okq)):
                pdq_pairs.append((a + s, b, int(dmin[a, b])))
            for a, b in zip(*np.nonzero(iu & (dmin <= NEAR_PDQ) & okq & (S >= NEAR_COS))):
                near.append((a + s, b))
    ti, tj, tc = np.concatenate(top_i), np.concatenate(top_j), np.concatenate(top_c)
    o = np.argsort(-tc)
    return {"pairs": n * (n - 1) // 2, "i": ti[o], "j": tj[o], "c": tc[o], "pdq_pairs": pdq_pairs, "near": near}


def region_pairs(H, src, keep_from, exclude):
    """Non-matching region queries: for each owner (a query photo), its best region cosine against each
    stored original (whole vector), except the owner's own parent/fillers (exclude(owner) -> set of j).
    src: dict owner, d. Returns the top scores (>= keep_from) and the number of (owner, stored) pairs."""
    owners = np.unique(src["owner"])
    top, pairs = [], 0
    for o in owners:
        sel = src["owner"] == o
        S = (src["d"][sel] @ H.D.T).max(axis=0)
        ex = exclude(o)
        if ex:
            S[list(ex)] = -1
        pairs += len(H.D) - len(ex)
        top.append(S[S >= keep_from])
    return np.concatenate(top) if top else np.zeros(0), pairs


# ---- candidates and the cascade ------------------------------------------------------------------

def whole_cands(H, Q, T, cfg, extras=None):
    """Per query: PDQ candidates [(j, dist)] (top 4) and DINOv2 candidates [(j, cos)] (top 4, >= T),
    plus the parent's PDQ distance and cosine."""
    out = []
    n = Q["n"]
    pd_par = np.zeros(n, np.int32)
    cos_par = np.zeros(n, np.float32)
    for s in range(0, n, 500):
        e = min(n, s + 500)
        S0 = Q["d"][s:e] @ H.D.T
        S = S0
        if extras:
            for c in extras:
                S = np.maximum(S, Q["d"][s:e] @ H.E[c].T)
        Dd = pdq_min_dist(Q["p8"][s:e], H.up)
        for r in range(e - s):
            g = s + r
            p = Q["parent"][g]
            pd_par[g], cos_par[g] = Dd[r, p], S[r, p]
            pc = []
            if Q["q"][g] > cfg["minQuality"]:
                js = np.nonzero((Dd[r] <= cfg["maxDistance"]) & (H.PQ > cfg["minQuality"]))[0]
                js = js[np.argsort(Dd[r, js], kind="stable")][:MAX_VERIFY_PER_STAGE]
                pc = [(int(j), int(Dd[r, j])) for j in js]
            js = np.nonzero(S[r] >= T)[0]
            js = js[np.argsort(-S[r, js], kind="stable")][:MAX_VERIFY_PER_STAGE]
            near = [int(j) for j, dd in pc if dd <= NEAR_PDQ and H.PQ[j] > cfg["minQuality"] and S0[r, j] >= NEAR_COS]
            out.append({"pdq": pc, "dino": [(int(j), float(S[r, j])) for j in js], "near": near})
    return out, pd_par, cos_par


def region_cands(H, Q, T, cfg, kinds):
    """Per query (owner): candidates from its regions of the given kinds: [(j, box, method)], at most 4
    per stage over all its regions (PDQ nearest first, then DINOv2 highest first), best region per stored photo."""
    R = Q["reg"]
    sel = np.nonzero(np.isin(R["kind"], [KINDS[k] for k in kinds]))[0]
    out = collections.defaultdict(list)
    if not len(sel):
        return out
    d, p8, q, own, box = R["d"][sel], R["p8"][sel], R["q"][sel], R["owner"][sel], R["box"][sel]
    for s in range(0, len(sel), 1000):
        e = min(len(sel), s + 1000)
        S = d[s:e] @ H.D.T
        Dd = pdq_min_dist(p8[s:e], H.up)
        for r in range(e - s):
            g = s + r
            if q[g] > cfg["minQuality"]:
                js = np.nonzero((Dd[r] <= cfg["maxDistance"]) & (H.PQ > cfg["minQuality"]))[0]
                for j in js:
                    out[int(own[g])].append(("pdq", int(j), float(Dd[r, j]), tuple(int(v) for v in box[g])))
            js = np.nonzero(S[r] >= T)[0]
            for j in js:
                out[int(own[g])].append(("dino", int(j), float(S[r, j]), tuple(int(v) for v in box[g])))
    final = {}
    for o, lst in out.items():
        pick = {}
        for stage, key in (("pdq", lambda x: x[2]), ("dino", lambda x: -x[2])):
            best = {}
            for x in sorted([x for x in lst if x[0] == stage], key=key):
                best.setdefault(x[1], x)
            for x in list(best.values())[:MAX_VERIFY_PER_STAGE]:
                pick.setdefault(x[1], x)
        final[o] = list(pick.values())
    return final


# ---- stage 3: the choice (calibrate half only) ---------------------------------------------------

def stage_calibrate():
    dest = os.path.join(CACHE, "calibrate.json")
    if os.path.exists(dest):
        return
    cfg = load_config()
    H = Half("calibrate")
    out = {}
    log("calibrate: originals pairs")
    scan = pair_scan(H, 0.3, cfg)
    out["whole"] = levels(scan["c"], scan["pairs"])
    out["pdq_pairs"] = len(scan["pdq_pairs"])
    out["near_pairs"] = len(scan["near"])
    for name, ex in EXTRA_SETS.items():
        sc = pair_scan(H, 0.3, cfg, extras=ex)
        out[f"whole{name}"] = levels(sc["c"], sc["pairs"])
    # copies against the originals that are not theirs (directed, correlated with the pairs above)
    top, pairs = [], 0
    for kind in TYPES:
        Q = H.queries[kind]
        S = Q["d"] @ H.D.T
        for r in range(Q["n"]):
            ex = {Q["parent"][r]} | {f for f in Q["fill"][r] if f >= 0}
            S[r, list(ex)] = -1
            pairs += len(H.D) - len(ex)
        top.append(S[S >= 0.3])
    out["copies_wrong"] = levels(np.concatenate(top), pairs)
    # regions: tiles and rectangles of ordinary photos, then of the copies (fillers and parent excluded)
    log("calibrate: region pairs")
    tops, pairs = [], 0
    for kinds, label in ((("tile",), "tiles"), (("new", "both"), "rects"), (("tile", "new", "both"), "regions")):
        sel = np.isin(H.oreg["kind"], [KINDS[k] for k in kinds])
        src = {"owner": H.oreg["owner"][sel], "d": H.oreg["d"][sel]}
        t_o, p_o = region_pairs(H, src, 0.3, lambda o: {o})
        t_c, p_c = [], 0
        for kind in REGION_TYPES:
            Q = H.queries[kind]
            sel = np.isin(Q["reg"]["kind"], [KINDS[k] for k in kinds])
            src = {"owner": Q["reg"]["owner"][sel], "d": Q["reg"]["d"][sel]}
            t, p = region_pairs(H, src, 0.3, lambda o, Q=Q: {Q["parent"][o]} | {f for f in Q["fill"][o] if f >= 0})
            t_c.append(t); p_c += p
        out[f"{label}_ordinary"] = levels(t_o, p_o) if len(t_o) > 20 else None
        allt = np.concatenate([t_o] + t_c)
        out[f"{label}_all"] = levels(allt, p_o + p_c)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    log(f"  saved {dest}")


# ---- verify plumbing -----------------------------------------------------------------------------

def vkey(qspec, j_spec, box):
    return json.dumps([list(qspec), list(j_spec), None if box is None else list(box)])


def load_vcache(name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {k: tuple(v) for k, v in json.load(f).items()}


def run_verifies(jobs, cache_name, label):
    """jobs: set of (qspec, cspec, box). Grouped by candidate (its SIFT computed once), cached."""
    done = load_vcache(cache_name)
    todo = [j for j in jobs if vkey(*j) not in done]
    if todo:
        log(f"{label}: {len(todo)} /verify calls ({len(done)} cached)")
        by = collections.defaultdict(list)
        for q, c, b in todo:
            by[c].append((q, b))
        items = sorted(by.items(), key=lambda kv: -len(kv[1]))
        # split long lists so the pool stays busy
        flat = []
        for c, lst in items:
            for s in range(0, len(lst), 24):
                flat.append((c, lst[s:s + 24]))
        # in batches, saved after each one (a restart of the computer loses one batch at most)
        for s0 in range(0, len(flat), 400):
            batch = flat[s0:s0 + 400]
            res = pool_map(_verify_worker, batch, f"{label} {s0}/{len(flat)}", chunk=1)
            for (c, lst), r in zip(batch, res):
                for (q, b), (n, ms) in zip(lst, r):
                    done[vkey(q, c, b)] = (n, ms)
            tmp = os.path.join(CACHE, cache_name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(done, f)
            os.replace(tmp, os.path.join(CACHE, cache_name))
    return done


# ---- stage 4: /verify on the calibrate half: K ---------------------------------------------------

def stage_calverify():
    dest = os.path.join(CACHE, "calverify.json")
    if os.path.exists(dest):
        return
    H = Half("calibrate")
    cfg = load_config()
    rng = np.random.default_rng(SEED)
    pos, neg = [], []
    per = VERIFY_POS // len(TYPES)
    for kind in TYPES:
        Q = H.queries[kind]
        for r in rng.choice(Q["n"], min(per, Q["n"]), replace=False):
            pos.append((kind, H.qspec(kind, int(r)), H.cspec(int(Q["parent"][r])), None))
    # region positives: the region of each screenshot / collage nearest its parent (new rectangles and tiles)
    for kind in REGION_TYPES:
        Q = H.queries[kind]
        R = Q["reg"]
        sel = np.nonzero(np.isin(R["kind"], [KINDS["new"], KINDS["both"], KINDS["tile"]]))[0]
        cos = np.einsum("ij,ij->i", R["d"][sel], H.D[Q["parent"][R["owner"][sel]]])
        best = {}
        for s, c in zip(sel, cos):
            o = int(R["owner"][s])
            if o not in best or c > best[o][1]:
                best[o] = (s, c)
        for o in rng.choice(sorted(best), min(250, len(best)), replace=False):
            s = best[int(o)][0]
            pos.append((kind + ":region", H.qspec(kind, int(o)), H.cspec(int(Q["parent"][o])), tuple(int(v) for v in R["box"][s])))
    scan = pair_scan(H, 0.3, cfg)
    for a, b, c in zip(scan["i"][:VERIFY_NEG // 2], scan["j"][:VERIFY_NEG // 2], scan["c"][:VERIFY_NEG // 2]):
        neg.append(("orig-orig", H.cspec(int(a)), H.cspec(int(b)), None, float(c)))
    per_neg = (VERIFY_NEG // 2) // len(TYPES)
    for kind in TYPES:
        Q = H.queries[kind]
        S = Q["d"] @ H.D.T
        for r in range(Q["n"]):
            S[r, [Q["parent"][r]] + [f for f in Q["fill"][r] if f >= 0]] = -1
        best = S.max(axis=1)
        jbest = S.argmax(axis=1)
        for r in np.argsort(-best)[:per_neg]:
            neg.append(("copy-wrong:" + kind, H.qspec(kind, int(r)), H.cspec(int(jbest[r])), None, float(best[r])))
    # region hard negatives: tiles and rectangles (ordinary photos and copies) against their most similar wrong original
    rn = []
    sel = np.isin(H.oreg["kind"], [KINDS["tile"], KINDS["new"], KINDS["both"]])
    S = H.oreg["d"][sel] @ H.D.T
    own = H.oreg["owner"][sel]
    S[np.arange(len(own)), own] = -1
    for k, s in zip(np.nonzero(sel)[0], range(len(own))):
        j = int(S[s].argmax())
        rn.append((float(S[s, j]), H.cspec(int(own[s])), H.cspec(j), tuple(int(v) for v in H.oreg["box"][k]), "region-ordinary"))
    for kind in REGION_TYPES:
        Q = H.queries[kind]
        R = Q["reg"]
        sel = np.nonzero(np.isin(R["kind"], [KINDS["tile"], KINDS["new"], KINDS["both"]]))[0]
        S = R["d"][sel] @ H.D.T
        for s, k in enumerate(sel):
            o = int(R["owner"][k])
            S[s, [Q["parent"][o]] + [f for f in Q["fill"][o] if f >= 0]] = -1
            j = int(S[s].argmax())
            rn.append((float(S[s, j]), H.qspec(kind, o), H.cspec(j), tuple(int(v) for v in R["box"][k]), "region-" + kind))
    rn.sort(key=lambda x: -x[0])
    seen = set()
    for c, q, cs, b, lab in rn:
        if (q, cs) in seen:
            continue
        seen.add((q, cs))
        neg.append((lab, q, cs, b, c))
        if len(seen) >= REGION_NEG:
            break
    jobs = {(q, c, b) for _, q, c, b in pos} | {(q, c, b) for _, q, c, b, _ in neg}
    done = run_verifies(jobs, "verify-calibrate.json", "verify calibrate")
    out = {"pos": [(lab, done[vkey(q, c, b)][0]) for lab, q, c, b in pos],
           "neg": [(lab, done[vkey(q, c, b)][0], cos) for lab, q, c, b, cos in neg]}
    # replay check against verify.verify (production code), and its time per call
    sample = [(c, q, b) for _, q, c, b in pos[::40]] + [(c, q, b) for _, q, c, b, _ in neg[::40]]
    chk = pool_map(_verify_check_worker, sample, "verify check", chunk=1)
    out["check"] = [(a, b, ms) for a, b, ms in chk]
    with open(dest, "w") as f:
        json.dump(out, f)
    log(f"  saved {dest}")


def choose_k():
    with open(os.path.join(CACHE, "calverify.json")) as f:
        v = json.load(f)
    neg = np.array([n for _, n, _ in v["neg"]])
    pos = np.array([n for _, n in v["pos"]])
    return int(neg.max()) + 1, v, pos, neg


def values():
    """The proposed values, from the calibrate half only."""
    with open(os.path.join(CACHE, "calibrate.json")) as f:
        c = json.load(f)
    K, v, pos, neg = choose_k()

    def T_of(L):
        m = L["max"]
        g = L["gpd"]["q"].get("1e-07")
        cand = m if g is None else max(m, g)
        return round(math.floor(cand * 200 + 1) / 200, 3)  # next 0.005 above

    out = {"K": K, "T": T_of(c["whole"]), "T_region": T_of(c["regions_all"]),
           "T_extra": {name: T_of(c[f"whole{name}"]) for name in EXTRA_SETS}}
    out["T_levels"] = {f: c["whole"]["emp"][str(f)] for f in (1e-4, 1e-5, 1e-6)}
    out["T_region_levels"] = {f: c["regions_all"]["emp"][str(f)] for f in (1e-4, 1e-5, 1e-6)}
    return out


# ---- stage 5: measure ------------------------------------------------------------------------------

POLICIES = ["whole", "rects_old", "rects_new", "tiles", "rects_new+tiles", "oracle"]


def measure_plan(H, V, cfg):
    """Candidates of every query under every evaluated setting, and the /verify jobs they need."""
    Ts = sorted({V["T"], *[V["T_levels"][f] for f in (1e-5, 1e-6) if V["T_levels"][f] is not None]})
    T_low = min(Ts)
    TR_ = [V["T_region"]]
    TR_low = V["T_region"]
    plan = {"Ts": Ts, "TRs": TR_, "types": {}}
    jobs = set()
    for kind, Q in H.queries.items():
        wc, pd_par, cos_par = whole_cands(H, Q, T_low, cfg)
        ex = {}
        for name, crops in EXTRA_SETS.items():
            ex[name] = whole_cands(H, Q, V["T_extra"][name], cfg, extras=crops)
        rc = {}
        for pol, kinds in (("rects_old", ("old", "both")), ("rects_new", ("new", "both")), ("tiles", ("tile",)),
                           ("rects_new+tiles", ("new", "both", "tile")), ("oracle", ("oracle",))):
            rc[pol] = region_cands(H, Q, TR_low, cfg, kinds)
        # old finder: its boxes are the "old" kind plus those identical to new ones
        plan["types"][kind] = {"whole": wc, "pd_par": pd_par, "cos_par": cos_par, "extra": ex, "regions": rc}
        for r, c in enumerate(wc):
            for j in {j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]}:
                jobs.add((H.qspec(kind, r), H.cspec(j), None))
        for name in EXTRA_SETS:
            for r, c in enumerate(ex[name][0]):
                for j in {j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]}:
                    jobs.add((H.qspec(kind, r), H.cspec(j), None))
        for pol, d in rc.items():
            for o, lst in d.items():
                for stage, j, sc, box in lst:
                    jobs.add((H.qspec(kind, o), H.cspec(j), box))
    return plan, jobs


def tally(H, plan, done, V, cfg, kind, T, TR_, K, near=False, policy="whole", region_when="signs_or_none", extras=None):
    """Detection of one type under one setting: per query, found (its parent confirmed), and the
    confirmations of other photos (wrong) and of photos shown in the copy (fillers)."""
    Q = H.queries[kind]
    P = plan["types"][kind]
    wc = P["whole"] if extras is None else P["extra"][extras][0]
    Tw = T if extras is None else V["T_extra"][extras]
    n = Q["n"]
    found = np.zeros(n, bool)
    wrong = filler = calls = 0
    used_regions = np.zeros(n, bool)
    for r in range(n):
        par = Q["parent"][r]
        fill = Q["fill"][r]
        cands = {j for j, _ in wc[r]["pdq"]} | {j for j, c in wc[r]["dino"] if c >= Tw}
        conf = set()
        for j in cands:
            calls += 1
            if done[vkey(H.qspec(kind, r), H.cspec(j), None)][0] >= K:
                conf.add(j)
        if near:
            conf |= set(wc[r]["near"])
        if policy != "whole":
            signs = bool(Q["signs"][r])  # the copy's own signs: rectangles found, or collage cells
            if region_when == "always" or (region_when == "signs_or_none" and (signs or not conf)) or (region_when == "signs" and signs):
                used_regions[r] = True
                for stage, j, sc, box in P["regions"][policy].get(r, []):
                    if stage == "dino" and sc < TR_:
                        continue
                    calls += 1
                    if done[vkey(H.qspec(kind, r), H.cspec(j), box)][0] >= K:
                        conf.add(j)
        found[r] = par in conf
        for j in conf:
            if j == par:
                continue
            if j in fill:
                filler += 1
            else:
                wrong += 1
    return {"n": n, "found": float(found.mean()), "found_mask": found, "wrong": wrong, "filler": filler, "calls": calls / max(1, n),
            "regions_used": float(used_regions.mean())}


def stage_measure():
    dest = os.path.join(CACHE, "measure.json")
    if os.path.exists(dest):
        return
    cfg = load_config()
    V = values()
    H = Half("measure")
    log(f"measure: T = {V['T']}, T_region = {V['T_region']}, K = {V['K']}, extras {V['T_extra']}")
    plan, jobs = measure_plan(H, V, cfg)
    # non-matching pairs of the measure originals passing a stage at the lowest T, and the 1,000 most similar
    scan = pair_scan(H, 0.3, cfg)
    low = min(plan["Ts"])
    nm = [(int(a), int(b)) for a, b, c in zip(scan["i"], scan["j"], scan["c"]) if c >= low]
    nm += [(int(a), int(b)) for a, b, _ in scan["pdq_pairs"]]
    nm += [(int(a), int(b)) for a, b in zip(scan["i"][:1000], scan["j"][:1000])]
    nm = sorted(set(nm))
    for a, b in nm:
        jobs.add((H.cspec(a), H.cspec(b), None))
    done = run_verifies(jobs, "verify-measure.json", "verify measure")
    K, T, TR_ = V["K"], V["T"], V["T_region"]
    K12 = cfg["minInliers"]
    res = {"V": V, "Ts": plan["Ts"], "TRs": plan["TRs"], "types": {}, "nonmatch": {}}
    for kind in H.queries:
        Q = H.queries[kind]
        P = plan["types"][kind]
        pdq_alone = (Q["q"] > cfg["minQuality"]) & (H.PQ[Q["parent"]] > cfg["minQuality"]) & (P["pd_par"] <= cfg["maxDistance"])
        row = {"n": Q["n"], "pdq": float(pdq_alone.mean()), "dino": float((P["cos_par"] >= T).mean()),
               "cos_par_p50": float(np.median(P["cos_par"])), "pd_par_p50": float(np.median(P["pd_par"]))}
        for lab, kw in {
            "cascade": {}, "cascade_K12": {"K": K12},
            "near": {"near": True},
            "rects_old": {"policy": "rects_old"}, "rects_new": {"policy": "rects_new"}, "tiles": {"policy": "tiles"},
            "rects_new+tiles": {"policy": "rects_new+tiles"}, "oracle": {"policy": "oracle"},
            "rects_new+tiles+near": {"policy": "rects_new+tiles", "near": True},
            "rects_new+tiles_signs": {"policy": "rects_new+tiles", "region_when": "signs"},
            **{f"extra{nm_}": {"extras": nm_} for nm_ in EXTRA_SETS},
            **{f"all{nm_}": {"extras": nm_, "policy": "rects_new+tiles", "near": True} for nm_ in EXTRA_SETS},
        }.items():
            args = dict(K=K, near=False, policy="whole", region_when="signs_or_none", extras=None)
            args.update(kw)
            t = tally(H, plan, done, V, cfg, kind, T, TR_, **args)
            t.pop("found_mask")
            row[lab] = t
        for Tl in plan["Ts"]:
            t = tally(H, plan, done, V, cfg, kind, Tl, TR_, K)
            row[f"T={Tl:.4f}"] = {k: v for k, v in t.items() if k != "found_mask"}
        # tiles were computed on a sample of the non-region types: rates over that sample only
        if kind not in REGION_TYPES and kind != "reupload":
            sub = np.nonzero(Q["tiled"])[0]
            for lab, pol in (("tiles_sample", "tiles"), ("rects_new+tiles_sample", "rects_new+tiles")):
                t = tally(H, plan, done, V, cfg, kind, T, TR_, K, policy=pol, region_when="always")
                base = tally(H, plan, done, V, cfg, kind, T, TR_, K)
                row[lab] = {"n": int(len(sub)), "found": float(t["found_mask"][sub].mean()), "base": float(base["found_mask"][sub].mean())}
        # how often the copy would show signs (rectangles found / collage cells)
        if kind != "reupload":
            row["with_new_rects"] = float((Q["n_new"] > 0).mean())
            row["with_old_rects"] = float((Q["n_old"] > 0).mean())
            row["collage_cells"] = float((Q["cells"] >= 2).mean())
            row["ms"] = {k: float(np.median(v[v > 0])) if (v > 0).any() else 0.0 for k, v in Q["ms"].items()}
            row["ms_p95"] = {k: float(np.percentile(v[v > 0], 95)) if (v > 0).any() else 0.0 for k, v in Q["ms"].items()}
            row["signs"] = float(Q["signs"].mean())
        if kind in REGION_TYPES:
            R = Q["reg"]
            for kk, codes in (("new", ("new", "both")), ("old", ("old", "both")), ("oracle", ("oracle",))):
                sel = np.isin(R["kind"], [KINDS[c] for c in codes])
                row[f"cos_region_{kk}_p50"] = float(np.median(np.einsum("ij,ij->i", R["d"][sel], H.D[Q["parent"][R["owner"][sel]]]))) if sel.any() else None
            # rectangle precision: best IoU of a new rectangle with the photo's box
            ious_new, ious_old = [], []
            for o in range(Q["n"]):
                b = Q["info"][o]["box"]
                for codes, lst in ((("new", "both"), ious_new), (("old", "both"), ious_old)):
                    sel = (R["owner"] == o) & np.isin(R["kind"], [KINDS[c] for c in codes])
                    lst.append(max([iou(b, x) for x in R["box"][sel]], default=0.0))
            row["iou_new_p50"] = float(np.median(ious_new))
            row["iou_new_07"] = float(np.mean(np.array(ious_new) >= 0.7))
            row["iou_old_07"] = float(np.mean(np.array(ious_old) >= 0.7))
        res["types"][kind] = row
    # non-matching pairs of the measure originals
    for Tl in plan["Ts"]:
        cos_pass = {(int(a), int(b)) for a, b, c in zip(scan["i"], scan["j"], scan["c"]) if c >= Tl}
        pdq_pass = {(int(a), int(b)) for a, b, _ in scan["pdq_pairs"]}
        st = cos_pass | pdq_pass
        res["nonmatch"][f"{Tl:.4f}"] = {"cos": len(cos_pass), "pdq": len(pdq_pass), "stage": len(st),
                                        "conf_K": sum(done[vkey(H.cspec(a), H.cspec(b), None)][0] >= K for a, b in st),
                                        "conf_12": sum(done[vkey(H.cspec(a), H.cspec(b), None)][0] >= K12 for a, b in st)}
    top_inl = [done[vkey(H.cspec(int(a)), H.cspec(int(b)), None)][0] for a, b in zip(scan["i"][:1000], scan["j"][:1000])]
    res["nonmatch"]["top1000"] = {"max": int(max(top_inl)), "p99": float(np.percentile(top_inl, 99)), "ge_K": int(sum(x >= K for x in top_inl)),
                                  "ge_12": int(sum(x >= K12 for x in top_inl)), "cos_max": float(scan["c"][0]), "tail": [float(x) for x in scan["c"][:10]]}
    res["nonmatch"]["near"] = len(scan["near"])
    res["pairs"] = scan["pairs"]
    # region false matches among the measure half's ordinary photos (tiles and new rectangles vs other originals)
    for kinds, label in ((("tile",), "tiles"), (("new", "both"), "rects"), (("tile", "new", "both"), "regions")):
        sel = np.isin(H.oreg["kind"], [KINDS[k] for k in kinds])
        src = {"owner": H.oreg["owner"][sel], "d": H.oreg["d"][sel]}
        t, p = region_pairs(H, src, 0.3, lambda o: {o})
        res["nonmatch"][f"{label}_ordinary"] = {"pairs": p, "ge_T_region": int((t >= TR_).sum()), "ge_T": int((t >= T).sum()),
                                                "max": float(t.max()) if len(t) else None}
    # ordinary photos: how often the finder returns rectangles or collage cells (cost of the region search)
    O = H.O
    res["ordinary"] = {"with_new_rects": float((O["n_new"][H.gidx] > 0).mean()), "with_old_rects": float((O["n_old"][H.gidx] > 0).mean()),
                       "collage_cells": float((O["cells"][H.gidx] >= 2).mean()), "flat_p50": float(np.median(O["flat"][H.gidx])),
                       "ms": {k[3:]: float(np.median(O[k][H.gidx][O[k][H.gidx] > 0])) if (O[k][H.gidx] > 0).any() else 0.0
                              for k in O if k.startswith("ms_")},
                       "ms_p95": {k[3:]: float(np.percentile(O[k][H.gidx][O[k][H.gidx] > 0], 95)) if (O[k][H.gidx] > 0).any() else 0.0
                                  for k in O if k.startswith("ms_")}}
    vm = np.array([v[1] for v in done.values()])
    res["verify_ms"] = {"p50": float(np.median(vm)), "p95": float(np.percentile(vm, 95)), "calls": len(vm)}
    with open(dest, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"  saved {dest}")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0



# ---- stage 6: ABO again (catalogue stress test), from the phase 1 cache -------------------------------

def _abo_region_worker(item):
    from app import images, screen
    path, kind = item
    with open(path, "rb") as f:
        im = images.decode(f.read())
    w, h = im.size
    t = time.perf_counter()
    rects, flat, cells = screen.photo_rects_part(im)
    ms = (time.perf_counter() - t) * 1000
    boxes = [r["box"] for r in rects if ok_box(r["box"])]
    regs = [("new", r) for r in region_fp(im, boxes)] + [("tile", r) for r in region_fp(im, tile_boxes(w, h))]
    return regs, cells, ms


def stage_abo():
    dest = os.path.join(CACHE, "abo.json")
    if os.path.exists(dest):
        return
    cfg = load_config()
    V = values()
    K, T, TR_ = V["K"], V["T"], V["T_region"]
    with open(os.path.join(HERE, "fingerprint-manifest.json")) as f:
        man = json.load(f)["images"]
    with open(os.path.join(ABO_FP, "transforms.json")) as f:
        trs = json.load(f)["transforms"]
    half_of = {x["id"]: x["half"] for x in man}
    tinfo = {t["id"]: t for t in trs}
    z = np.load(os.path.join(ABO_CACHE, "fp-originals.npz"))
    O = {k: z[k] for k in z.files}
    out = {"V": V}
    dups = {}
    for half in ("calibrate", "measure"):
        dz = np.load(os.path.join(ABO_CACHE, f"dups-{half}.npz"))
        dups[half] = {frozenset((str(a), str(b))) for a, b, fl in zip(dz["a"], dz["b"], dz["flag"]) if fl}
    # near-identity rule among the originals of each half (raw pairs; duplicates counted apart)
    for half in ("calibrate", "measure"):
        idx = np.nonzero(np.array([half_of[i] == half for i in O["ids"]]))[0]
        D, Q8, PQ = f32(O["dino"][idx]), as_u64(O["pdq"][idx]), O["quality"][idx]
        n = len(idx)
        near, cos_hi = [], 0
        for s in range(0, n, 500):
            e = min(n, s + 500)
            S = D[s:e] @ D.T
            iu = np.arange(s, e)[:, None] < np.arange(n)[None, :]
            cos_hi += int((iu & (S >= T)).sum())
            dmin = np.minimum(pdq_min_dist(Q8[s:e], Q8[:, 0, :]), pdq_min_dist(Q8, Q8[s:e, 0, :]).T)
            okq = (PQ[s:e, None] > cfg["minQuality"]) & (PQ[None, :] > cfg["minQuality"])
            for a, b in zip(*np.nonzero(iu & (dmin <= NEAR_PDQ) & okq & (S >= NEAR_COS))):
                near.append((str(O["ids"][idx[a + s]]), str(O["ids"][idx[b]])))
        nd = sum(frozenset(p) in dups[half] for p in near)
        out[f"near_{half}"] = {"pairs": n * (n - 1) // 2, "near": len(near), "near_dup": nd, "near_other": len(near) - nd, "cos_ge_T": cos_hi,
                               "examples": near[:10]}
    # measure half: the cascade at the new values
    midx = np.nonzero(np.array([half_of[i] == "measure" for i in O["ids"]]))[0]
    mid = O["ids"][midx]
    pos_of = {str(i): k for k, i in enumerate(mid)}
    Dm, Q8m, PQm = f32(O["dino"][midx]), as_u64(O["pdq"][midx]), O["quality"][midx]

    class AH:  # the Half interface of whole_cands
        pass
    H = AH()
    H.D, H.up, H.PQ, H.E = Dm, Q8m[:, 0, :], PQm, {}
    opath = lambda i: os.path.join(ABO_FP, "originals", str(i) + ".jpg")
    queries = {"reupload": {"ids": mid, "parent": np.arange(len(mid)), "d": Dm, "p8": Q8m, "q": PQm, "path": [opath(i) for i in mid], "n": len(mid)}}
    for t in TYPES:
        F = np.load(os.path.join(ABO_CACHE, f"fp-{t}.npz"))
        rows = np.array([k for k, i in enumerate(F["ids"]) if tinfo[str(i)]["half"] == "measure"])
        ids = F["ids"][rows]
        queries[t] = {"ids": ids, "parent": np.array([pos_of[tinfo[str(i)]["parent"]] for i in ids]), "d": f32(F["dino"][rows]),
                      "p8": as_u64(F["pdq"][rows]), "q": F["quality"][rows], "path": [os.path.join(ART, tinfo[str(i)]["file"]) for i in ids],
                      "n": len(rows), "fill": [{pos_of[f] for f in tinfo[str(i)].get("fillers") or []} for i in ids]}
    jobs = set()
    cands = {}
    for name, Q in queries.items():
        wc, pdp, cosp = whole_cands(H, Q, T, cfg)
        cands[name] = (wc, pdp, cosp)
        for r, c in enumerate(wc):
            for j in {j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]}:
                jobs.add((("abo", Q["path"][r]), ("abo", opath(mid[j])), None))
    # regions of a seeded sample of the screenshots and collages (new finder and tiles)
    rsamp = {}
    for t in REGION_TYPES:
        Q = queries[t]
        sub = seeded(Q["n"], 1000, f"abo-regions-{t}")
        rp = os.path.join(CACHE, f"abo-regions-{t}.npz")
        if not os.path.exists(rp):
            res = pool_map(_abo_region_worker, [(Q["path"][r], t) for r in sub], f"abo regions {t}")
            regs = [(k, KINDS[kk], b, p8, q, d) for k, (rr, _, _) in enumerate(res) for kk, (b, p8, q, d) in rr]
            np.savez(rp, sub=sub, cells=np.array([r[1] for r in res]), ms=np.array([r[2] for r in res], np.float32),
                     r_owner=np.array([x[0] for x in regs], np.int64), r_kind=np.array([x[1] for x in regs], np.int8),
                     r_box=np.array([x[2] for x in regs], np.int32).reshape(-1, 4), r_pdq=np.array([x[3] for x in regs], np.uint8).reshape(-1, 8, 32),
                     r_q=np.array([x[4] for x in regs], np.int16), r_dino=np.array([x[5] for x in regs], "<f2").reshape(-1, 384))
        zz = np.load(rp)
        RQ = {"reg": {"owner": zz["r_owner"], "kind": zz["r_kind"], "box": zz["r_box"], "d": f32(zz["r_dino"]), "p8": as_u64(zz["r_pdq"]), "q": zz["r_q"]}}
        rc = region_cands(H, RQ, TR_, cfg, ("new", "tile"))
        rsamp[t] = (zz["sub"], rc, zz["cells"])
        for o, lst in rc.items():
            r = int(zz["sub"][o])
            for stage, j, sc, box in lst:
                jobs.add((("abo", Q["path"][r]), ("abo", opath(mid[j])), box))
    # non-matching pairs of the measure originals that pass a stage
    nm = []
    for s in range(0, len(mid), 500):
        e = min(len(mid), s + 500)
        S = Dm[s:e] @ Dm.T
        iu = np.arange(s, e)[:, None] < np.arange(len(mid))[None, :]
        dmin = np.minimum(pdq_min_dist(Q8m[s:e], Q8m[:, 0, :]), pdq_min_dist(Q8m, Q8m[s:e, 0, :]).T)
        okq = (PQm[s:e, None] > cfg["minQuality"]) & (PQm[None, :] > cfg["minQuality"])
        for a, b in zip(*np.nonzero(iu & ((S >= T) | ((dmin <= cfg["maxDistance"]) & okq)))):
            nm.append((a + s, b))
    for a, b in nm:
        jobs.add((("abo", opath(mid[a])), ("abo", opath(mid[b])), None))
    # /verify changed since phase 1 (degenerate homographies, mirrored try): every pair is verified again.
    done = run_verifies(jobs, "verify-abo.json", "verify abo")
    mdups = dups["measure"]
    isdup = lambda a, b: frozenset((str(mid[a]), str(mid[b]))) in mdups
    rows = {}
    for name, Q in queries.items():
        wc, pdp, cosp = cands[name]
        n = Q["n"]
        row = {"n": n, "pdq": float(((Q["q"] > cfg["minQuality"]) & (PQm[Q["parent"]] > cfg["minQuality"]) & (pdp <= cfg["maxDistance"])).mean()),
               "dino": float((cosp >= T).mean())}
        for lab, near in (("cascade", False), ("near", True)):
            found = np.zeros(n, bool)
            wrong = dupc = fill = 0
            for r, c in enumerate(wc):
                par = Q["parent"][r]
                conf = {j for j in ({j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]})
                        if done[vkey(("abo", Q["path"][r]), ("abo", opath(mid[j])), None)][0] >= K}
                if near:
                    conf |= set(c["near"])
                found[r] = par in conf
                for j in conf - {par}:
                    if j in Q.get("fill", [set()] * n)[r]:
                        fill += 1
                    elif isdup(par, j):
                        dupc += 1
                    else:
                        wrong += 1
            row[lab] = {"found": float(found.mean()), "wrong": wrong, "dup": dupc, "filler": fill}
            if name in rsamp and not near:
                sub, rc, cells = rsamp[name]
                f2 = found[sub].copy()
                w2 = 0
                for o, lst in rc.items():
                    r = int(sub[o])
                    for stage, j, sc, box in lst:
                        if done[vkey(("abo", Q["path"][r]), ("abo", opath(mid[j])), box)][0] >= K:
                            if j == Q["parent"][r]:
                                f2[o] = True
                            elif j not in Q["fill"][r] and not isdup(Q["parent"][r], j):
                                w2 += 1
                row["regions_sample"] = {"n": int(len(sub)), "whole": float(found[sub].mean()), "with_regions": float(f2.mean()), "wrong": w2,
                                         "cells": float((cells >= 2).mean())}
        rows[name] = row
    out["types"] = rows
    st = {"pairs": len(mid) * (len(mid) - 1) // 2, "stage": len(nm)}
    conf = [(a, b) for a, b in nm if done[vkey(("abo", opath(mid[a])), ("abo", opath(mid[b])), None)][0] >= K]
    st["conf"] = len(conf)
    st["conf_dup"] = sum(isdup(a, b) for a, b in conf)
    out["nonmatch"] = st
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, default=float)
    log(f"  saved {dest}")


# ---- stage 7: the stored centre crop with the 1e-5 level (the 70 % crop target), measure half -------

def stage_loose():
    dest = os.path.join(CACHE, "loose.json")
    if os.path.exists(dest):
        return
    cfg = load_config()
    V = values()
    with open(os.path.join(CACHE, "calibrate.json")) as f:
        C = json.load(f)
    T = float(C["whole+c70"]["emp"]["1e-05"])  # chosen on the calibrate half
    K = V["K"]
    H = Half("measure")
    plan = {}
    jobs = set()
    for kind, Q in H.queries.items():
        wc, _, cos_par = whole_cands(H, Q, T, cfg, extras=["c70"])
        plan[kind] = wc
        for r, c in enumerate(wc):
            for j in {j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]}:
                jobs.add((H.qspec(kind, r), H.cspec(j), None))
    sc = pair_scan(H, T - 0.001, cfg, extras=["c70"])
    nm = [(int(a), int(b)) for a, b, c in zip(sc["i"], sc["j"], sc["c"]) if c >= T]
    for a, b in nm:
        jobs.add((H.cspec(a), H.cspec(b), None))
    done = run_verifies(jobs, "verify-measure.json", "verify loose")
    out = {"T": T, "K": K, "types": {}, "nonmatch": {"pairs": sc["pairs"], "ge_T": len(nm),
           "conf": sum(done[vkey(H.cspec(a), H.cspec(b), None)][0] >= K for a, b in nm)}}
    for kind, Q in H.queries.items():
        found = wrong = 0
        for r, c in enumerate(plan[kind]):
            conf = {j for j in ({j for j, _ in c["pdq"]} | {j for j, _ in c["dino"]}) if done[vkey(H.qspec(kind, r), H.cspec(j), None)][0] >= K}
            found += Q["parent"][r] in conf
            wrong += len([j for j in conf if j != Q["parent"][r] and j not in Q["fill"][r]])
        out["types"][kind] = {"n": Q["n"], "found": found / Q["n"], "wrong": wrong}
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


# ---- tables for the report ------------------------------------------------------------------------

def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def stage_report():
    out = []
    w = out.append
    with open(os.path.join(CACHE, "calibrate.json")) as f:
        C = json.load(f)
    K, v, pos, neg = choose_k()
    V = values()
    w(f"values: {json.dumps(V)}")
    w("\n## DINOv2 levels (calibrate half)\n")
    w("| Pair set | Pairs | 1e-4 | 1e-5 | 1e-6 | 1e-7 | max | GPD 1e-6 | GPD 1e-7 | GPD end |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ["whole", "copies_wrong"] + [f"whole{n}" for n in EXTRA_SETS] + ["tiles_ordinary", "rects_ordinary", "regions_ordinary", "tiles_all", "rects_all", "regions_all"]:
        L = C.get(key)
        if not L:
            continue
        e, g = L["emp"], L["gpd"]
        f = lambda x: "—" if x is None else f"{x:.4f}"
        w(f"| {key} | {L['pairs']:,} | {f(e.get('0.0001'))} | {f(e.get('1e-05'))} | {f(e.get('1e-06'))} | {f(e.get('1e-07'))} | {L['max']:.4f} | "
          f"{f(g['q'].get('1e-06'))} | {f(g['q'].get('1e-07'))} | {f(g.get('endpoint'))} |")
        w(f"<!-- {key} tail: {', '.join(f'{x:.4f}' for x in L['tail'][:10])} -->")
    w(f"\nPDQ pairs (<= 31, q > 49) among calibrate originals: {C['pdq_pairs']}; near-identity pairs: {C['near_pairs']}")
    w("\n## /verify (calibrate half)\n")
    labs = collections.defaultdict(list)
    for lab, n in v["pos"]:
        labs[lab].append(n)
    w("| Positives | Pairs | p5 | p50 | >= 12 | >= K |")
    w("|---|---:|---:|---:|---:|---:|")
    for lab, xs in sorted(labs.items()):
        xs = np.array(xs)
        w(f"| {lab} | {len(xs)} | {np.percentile(xs, 5):.0f} | {np.percentile(xs, 50):.0f} | {pct((xs >= 12).mean())} | {pct((xs >= K).mean())} |")
    w(f"| all | {len(pos)} | {np.percentile(pos, 5):.0f} | {np.percentile(pos, 50):.0f} | {pct((pos >= 12).mean())} | {pct((pos >= K).mean())} |")
    nl = collections.defaultdict(list)
    for lab, n, c in v["neg"]:
        nl[lab.split(":")[0]].append((n, c))
    w("\n| Hard negatives | Pairs | cosine range | inliers p50 | p99 | max | >= 12 |")
    w("|---|---:|---|---:|---:|---:|---:|")
    for lab, xs in sorted(nl.items()):
        n = np.array([x[0] for x in xs]); c = np.array([x[1] for x in xs])
        w(f"| {lab} | {len(n)} | {c.min():.3f} – {c.max():.3f} | {np.median(n):.0f} | {np.percentile(n, 99):.0f} | {n.max()} | {(n >= 12).sum()} |")
    w("\n| K | hard negatives confirmed | positives found |")
    w("|---:|---:|---:|")
    for k in sorted({8, 10, 12, 14, 16, 20, 25, 30, K}):
        w(f"| {k} | {(neg >= k).sum()} | {pct((pos >= k).mean())} |")
    chk = np.array(v["check"])
    w(f"\nreplay check: {int((chk[:, 0] == chk[:, 1]).sum())}/{len(chk)} equal; verify.verify p50 {np.median(chk[:, 2]):.0f} ms, p95 {np.percentile(chk[:, 2], 95):.0f} ms")
    p = os.path.join(CACHE, "measure.json")
    if os.path.exists(p):
        with open(p) as f:
            M = json.load(f)
        w("\n## Detection, measure half\n")
        cols = ["pdq", "dino", "cascade", "cascade_K12", "near", "rects_old", "rects_new", "tiles", "rects_new+tiles", "rects_new+tiles+near",
                "rects_new+tiles_signs", "oracle"] + [f"extra{n}" for n in EXTRA_SETS] + [f"all{n}" for n in EXTRA_SETS]
        w("| Type | n | " + " | ".join(cols) + " |")
        w("|---|---:|" + "---:|" * len(cols))
        tot = collections.defaultdict(float)
        for kind, row in M["types"].items():
            cells = []
            for c in cols:
                x = row.get(c)
                val = x if isinstance(x, (int, float)) else (x or {}).get("found")
                cells.append(pct(val))
                if val is not None:
                    tot[c] += val * row["n"]
            w(f"| {kind} | {row['n']} | " + " | ".join(cells) + " |")
        n9 = sum(r["n"] for k, r in M["types"].items() if k != "reupload")
        w("\n| Type | wrong confirmed (cascade / +regions / +regions+near / all+c70) | fillers (+regions) | /verify calls per photo (cascade / +regions) | regions used |")
        w("|---|---|---:|---|---:|")
        for kind, row in M["types"].items():
            w(f"| {kind} | {row['cascade']['wrong']} / {row['rects_new+tiles']['wrong']} / {row['rects_new+tiles+near']['wrong']} / {row['all+c70']['wrong']} | "
              f"{row['rects_new+tiles']['filler']} | {row['cascade']['calls']:.2f} / {row['rects_new+tiles']['calls']:.2f} | {pct(row['rects_new+tiles']['regions_used'])} |")
        w("\n| Type | T levels: found (wrong) |")
        w("|---|---|")
        for kind, row in M["types"].items():
            w(f"| {kind} | " + "; ".join(f"{k}: {pct(x['found'])} ({x['wrong']})" for k, x in row.items() if k.startswith("T=")) + " |")
        w("\n| Type | tiles sample n | base | +tiles | +rects+tiles |")
        w("|---|---:|---:|---:|---:|")
        for kind, row in M["types"].items():
            if "tiles_sample" in row:
                w(f"| {kind} | {row['tiles_sample']['n']} | {pct(row['tiles_sample']['base'])} | {pct(row['tiles_sample']['found'])} | {pct(row['rects_new+tiles_sample']['found'])} |")
        w("\n| Region type | IoU>=0.7 new | old | cos parent p50 new / old / oracle | signs | cells>=2 |")
        w("|---|---:|---:|---|---:|---:|")
        for kind in REGION_TYPES:
            row = M["types"][kind]
            f = lambda x: "—" if x is None else f"{x:.3f}"
            w(f"| {kind} | {pct(row['iou_new_07'])} | {pct(row['iou_old_07'])} | {f(row['cos_region_new_p50'])} / {f(row['cos_region_old_p50'])} / {f(row['cos_region_oracle_p50'])} | {pct(row['signs'])} | {pct(row['collage_cells'])} |")
        w("\n| Copy type | ms p50: make, decode, fp, rects new, rects old, tiles, rect fp | p95 fp, rects new, tiles |")
        w("|---|---|---|")
        for kind, row in M["types"].items():
            if "ms" in row:
                m, q = row["ms"], row["ms_p95"]
                w(f"| {kind} | {m['make']:.0f}, {m['decode']:.0f}, {m['fp']:.0f}, {m['rects_new']:.0f}, {m['rects_old']:.0f}, {m['tiles']:.0f}, {m['rect_fp']:.0f} | {q['fp']:.0f}, {q['rects_new']:.0f}, {q['tiles']:.0f} |")
        w(f"\nordinary (measure originals): {json.dumps(M['ordinary'])}")
        w(f"\nnonmatch: {json.dumps(M['nonmatch'])}")
        w(f"\nverify ms: {json.dumps(M['verify_ms'])}; pairs {M['pairs']}")
    p = os.path.join(CACHE, "abo.json")
    if os.path.exists(p):
        with open(p) as f:
            A = json.load(f)
        w("\n## ABO stress test\n")
        w("| Type | n | PDQ | DINOv2 | cascade | wrong / dup / filler | +near | regions sample (n: whole -> +regions, wrong) |")
        w("|---|---:|---:|---:|---:|---|---:|---|")
        for kind, row in A["types"].items():
            rs = row.get("regions_sample")
            rtxt = "" if not rs else f"{rs['n']}: {pct(rs['whole'])} -> {pct(rs['with_regions'])}, {rs['wrong']}"
            c = row["cascade"]
            w(f"| {kind} | {row['n']} | {pct(row['pdq'])} | {pct(row['dino'])} | {pct(c['found'])} | {c['wrong']} / {c['dup']} / {c['filler']} | {pct(row['near']['found'])} | {rtxt} |")
        w(f"\nnonmatch: {json.dumps(A['nonmatch'])}")
        w(f"near: {json.dumps({k: {kk: vv for kk, vv in x.items() if kk != 'examples'} for k, x in A.items() if k.startswith('near_')})}")
    O = load_npz("originals.npz")
    w("\n## Originals (one thread)\n")
    for k in O:
        if k.startswith("ms_"):
            x = O[k][O[k] > 0]
            if len(x):
                w(f"- {k[3:]}: p50 {np.median(x):.0f} ms, p95 {np.percentile(x, 95):.0f} ms (n {len(x)})")
    w(f"- with new rects {pct((O['n_new'] > 0).mean())}, old {pct((O['n_old'] > 0).mean())}, collage cells {pct((O['cells'] >= 2).mean())}")
    txt = "\n".join(out)
    with open(os.path.join(CACHE, "report-tables.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


STAGES = ["originals", "queries", "calibrate", "calverify", "measure", "abo"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(CACHE, exist_ok=True)
    for st in STAGES:
        if what in ("all", st):
            globals()[f"stage_{st}"]()
    if what == "values":
        print(json.dumps(values(), indent=1))
    if what in ("all", "loose"):
        stage_loose()
    if what in ("all", "report"):
        stage_report()
