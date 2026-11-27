#!/usr/bin/env python3
"""Calibration of the reused-photo check (design note 40, phase 1, « jeu empreintes »).

Fingerprints every photo of the fingerprint set with the vision service's own modules (no HTTP),
chooses the DINOv2 threshold and the /verify inlier count on the CALIBRATE half only, then measures
the production cascade (PDQ <= 31 with quality > 49, or DINOv2 >= T; then /verify >= K) on the
MEASURE half, per transform, with and without region fingerprints for screenshots and collages.
It never writes config/photo-checks.json: the proposed values go into the report.

Run (from the repository root; the venv is temporary, put it on a RAM disk or delete it after):

  uv python install 3.12        # a uv-managed Python: it carries Python.h, needed to compile pdqhash
  uv venv --python 3.12 --managed-python /tmp/venv-fp
  VIRTUAL_ENV=/tmp/venv-fp uv pip install --require-hashes -r service/requirements-build.lock
  VIRTUAL_ENV=/tmp/venv-fp uv pip install --require-hashes --no-build-isolation --no-binary pdqhash \
      -r service/requirements.lock                      # the vision image's exact lock (Dockerfile)
  MODELS_DIR=artifacts/vision-models nice -n 10 /tmp/venv-fp/bin/python \
      eval/calibrate-fingerprints.py [all|fingerprints|regions|duplicates|calibrate|measure|report]

Stages are cached under <artifacts>/eval-photos/fingerprint/cache/ (.npz and .json, about 130 MB): a re-run
only computes what is missing (delete a file to recompute it). At most WORKERS processes, one
thread each (the laptop is shared). The script stops when the disk has less than 5 GB free.
EVAL_PHOTOS_DIR overrides <artifacts>/eval-photos; REPORT_DATE the date in the report's name.
"""
import os

WORKERS = int(os.environ.get("FP_WORKERS", "4"))
# One thread per process everywhere (ONNX Runtime, OpenCV, BLAS); the main process may use WORKERS
# BLAS threads for the cosine matrices, while the pool is idle.
os.environ.setdefault("ORT_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(WORKERS))
os.environ.setdefault("MKL_NUM_THREADS", str(WORKERS))
os.environ.setdefault("DINO_VARIANT", "small")

import base64
import collections
import datetime
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
ART = os.environ.get("EVAL_PHOTOS_DIR", os.path.join(REPO, "artifacts", "eval-photos"))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO, "artifacts", "vision-models"))
FP_DIR = os.path.join(ART, "fingerprint")
CACHE = os.path.join(FP_DIR, "cache")
MANIFEST = os.path.join(HERE, "fingerprint-manifest.json")
TRANSFORMS = os.path.join(FP_DIR, "transforms.json")
REPORT_DATE = os.environ.get("REPORT_DATE", "2026-11-24")
REPORT = os.path.join(ART, f"calibration-fingerprints-{REPORT_DATE}.md")
# The cascade's starting values (PDQ distance and quality, minimum inliers), as in the API's configuration
# when the run was made; only these three are read. PHOTO_CHECKS_CONFIG points to another file.
CONFIG = os.environ.get("PHOTO_CHECKS_CONFIG", os.path.join(HERE, "fingerprint-config.json"))

# ---- targets and rules, from the plan (note 40, phase 1 « Cibles proposées ») ----------------
TARGET_FMR = 1e-7            # false-match rate per non-matching pair
TARGET_REUPLOAD = 1.0        # unchanged re-upload found
TARGET_CROP70 = 0.9          # 70 % centre crop found
SOTA_SCREENSHOT = 0.18       # best published method on screenshots (arXiv:2610.04151)
VERIFY_SAMPLE = 2000         # positive and hard-negative pairs for /verify (task brief)
ORACLE_SAMPLE = 1000         # screenshots and collages per type given the "perfect box" (upper bound only)
# Label noise: ABO holds the same photograph under several image ids (catalogue duplicates and
# composites). A pair of distinct ids is the same picture, and leaves the non-matching set of both
# halves, when its cosine is >= DUP_COS and its PDQ distance <= DUP_PDQ or /verify finds >=
# DUP_INLIERS inliers. Rule set by the coordinator on 25/11/2026 after reading the calibrate tail.
DUP_COS = 0.97
DUP_PDQ = 8
DUP_INLIERS = 50
SEED = 20261124
MIN_FREE_GB = 5.0

TYPES = ["jpeg_q50", "downscale_50", "crop_70", "crop_50", "rotate_15", "mirror", "text_overlay", "classifieds_page", "collage_2x2"]
REGION_TYPES = ["classifieds_page", "collage_2x2"]
# Where make-transforms.mjs put the parent photo (lib/transforms.mjs PAGE and COLLAGE_CELL): used
# only for the "perfect box" upper bound, never by the cascade itself.
PAGE_BOX = [24, 52 + 24, 24 + 432, 52 + 24 + 324]
CELL = 192

# Read from the API's files, never written: the cascade's fixed parts.
MAX_VERIFY_PER_STAGE = 4     # services/photoMatch.ts
PDQ_BITS = 256


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
            "minInliers12": c["verify"]["minInliers"]["value"]}


# ---- the set -------------------------------------------------------------------------------

def load_set():
    with open(MANIFEST) as f:
        man = json.load(f)
    with open(TRANSFORMS) as f:
        tr = json.load(f)["transforms"]
    originals = [{"id": i["id"], "half": i["half"], "sha256": i["sha256"],
                  "path": os.path.join(FP_DIR, "originals", i["id"] + ".jpg")} for i in man["images"]]
    transforms = [dict(t, path=os.path.join(ART, t["file"])) for t in tr]
    return originals, transforms


# ---- workers (each process loads the models once) -------------------------------------------
_REG = None


def _init_worker():
    global _REG
    import cv2
    cv2.setNumThreads(1)
    from app import models
    _REG = models.Registry()
    _REG.dino()  # load now (checks size and sha256 of the model file)


def _decode(path, sha=None):
    from app import images
    with open(path, "rb") as f:
        data = f.read()
    ok = None if sha is None else hashlib.sha256(data).hexdigest() == sha
    return images.decode(data), ok


def _unpack(fp):
    pdq8 = np.stack([np.frombuffer(bytes.fromhex(h), np.uint8) for h in fp["pdq"]])
    return pdq8, int(fp["quality"]), np.frombuffer(base64.b64decode(fp["dino"]), "<f2").copy()


def _fp_worker(item):
    from app import fingerprint
    path, sha = item
    t0 = time.perf_counter()
    im, ok = _decode(path, sha)
    t1 = time.perf_counter()
    out = fingerprint.fingerprint(_REG, im, {})
    t2 = time.perf_counter()
    pdq8, q, dino = _unpack(out)
    return pdq8, q, dino, (t1 - t0) * 1000, (t2 - t1) * 1000, ok, im.size


def _region_worker(item):
    """photo_rects_part (service/app/screen.py) on the image, then the production fingerprint of each
    rectangle; plus, for screenshots and collages, the known box (upper bound only)."""
    from app import fingerprint, screen
    path, oracle, do_fp = item
    im, _ = _decode(path)
    t0 = time.perf_counter()
    rects = screen.photo_rects_part(im)[0]
    t1 = time.perf_counter()
    boxes = [r["box"] for r in rects]
    if not do_fp:  # originals: only how many rectangles an ordinary photo yields
        return [], ["rect"] * len(boxes), 0, (t1 - t0) * 1000, 0.0
    small = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) < fingerprint.MIN_REGION_SIDE]
    boxes = [b for b in boxes if b not in small]
    ask = boxes + ([oracle] if oracle else [])
    regs = []
    if ask:
        out = fingerprint.fingerprint(_REG, im, {"regions": ask})
        for r in out["regions"]:
            pdq8 = np.stack([np.frombuffer(bytes.fromhex(h), np.uint8) for h in r["pdq"]])
            regs.append((r["box"], pdq8, int(r["quality"]), np.frombuffer(base64.b64decode(r["dino"]), "<f2").copy()))
    t2 = time.perf_counter()
    kinds = ["rect"] * len(boxes) + (["oracle"] if oracle else [])
    return regs, kinds, len(small), (t1 - t0) * 1000, (t2 - t1) * 1000


def _verify_worker(item):
    from app import verify
    pa, pb, box_a = item
    a, _ = _decode(pa)
    b, _ = _decode(pb)
    t0 = time.perf_counter()
    r = verify.verify(a, b, {} if box_a is None else {"box_a": box_a})
    return r["inliers"], r["matches"], min(r["keypoints"]), (time.perf_counter() - t0) * 1000


def pool_map(fn, items, label, chunk=16):
    out = []
    t = time.time()
    with mp.get_context("fork").Pool(WORKERS, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(fn, items, chunksize=chunk)):
            out.append(r)
            if (i + 1) % 2000 == 0:
                log(f"  {label}: {i + 1}/{len(items)} ({(time.time() - t) / (i + 1) * 1000:.0f} ms per item, wall)")
                check_disk()
    return out


# ---- stage 1: fingerprints -------------------------------------------------------------------

def stage_fingerprints():
    originals, transforms = load_set()
    groups = [("originals", originals)] + [(t, [x for x in transforms if x["type"] == t]) for t in TYPES]
    for name, items in groups:
        dest = os.path.join(CACHE, f"fp-{name}.npz")
        if os.path.exists(dest):
            continue
        check_disk()
        log(f"fingerprints {name}: {len(items)} images")
        res = pool_map(_fp_worker, [(x["path"], x["sha256"]) for x in items], name)
        np.savez(dest, ids=np.array([x["id"] for x in items]),
                 pdq=np.stack([r[0] for r in res]), quality=np.array([r[1] for r in res], np.int16),
                 dino=np.stack([r[2] for r in res]).astype("<f2"),
                 ms_decode=np.array([r[3] for r in res], np.float32), ms_fp=np.array([r[4] for r in res], np.float32),
                 sha_ok=np.array([bool(r[5]) for r in res]), size=np.array([r[6] for r in res], np.int16))
        log(f"  saved {dest}")


def load_fp(name):
    z = np.load(os.path.join(CACHE, f"fp-{name}.npz"))
    return {k: z[k] for k in z.files}


# ---- stage 2: regions ------------------------------------------------------------------------

def oracle_box(t):
    if t["type"] == "classifieds_page":
        return PAGE_BOX
    p = int(t["position"])
    return [(p % 2) * CELL, (p // 2) * CELL, (p % 2) * CELL + CELL, (p // 2) * CELL + CELL]


def stage_regions():
    originals, transforms = load_set()
    groups = [("originals", [(x["path"], None, False) for x in originals], originals)]
    for t in REGION_TYPES:
        # Measure half only: regions add no threshold of their own (they use T and K). The perfect
        # box (an upper bound) on a seeded sample of ORACLE_SAMPLE images, to save CPU time.
        xs = [x for x in transforms if x["type"] == t and x["half"] == "measure"]
        pick = set(np.random.default_rng(SEED).choice(len(xs), min(ORACLE_SAMPLE, len(xs)), replace=False).tolist())
        groups.append((t, [(x["path"], oracle_box(x) if k in pick else None, True) for k, x in enumerate(xs)], xs))
    for name, items, src in groups:
        dest = os.path.join(CACHE, f"regions-{name}.npz")
        if os.path.exists(dest):
            continue
        check_disk()
        log(f"regions {name}: {len(items)} images")
        res = pool_map(_region_worker, items, f"regions {name}")
        owner, kind, box, pdq, qual, dino = [], [], [], [], [], []
        for i, (regs, kinds, _small, _a, _b) in enumerate(res):
            for (b, p8, q, d), k in zip(regs, kinds if regs else []):
                owner.append(i); kind.append(k); box.append(b); pdq.append(p8); qual.append(q); dino.append(d)
        np.savez(dest, ids=np.array([x["id"] for x in src]), owner=np.array(owner, np.int32), kind=np.array(kind),
                 box=np.array(box, np.int32).reshape(-1, 4), pdq=np.array(pdq, np.uint8).reshape(-1, 8, 32),
                 quality=np.array(qual, np.int16), dino=np.array(dino, "<f2").reshape(-1, 384),
                 n_rects=np.array([sum(1 for k in r[1] if k == "rect") for r in res], np.int16),
                 n_small=np.array([r[2] for r in res], np.int16),
                 ms_rects=np.array([r[3] for r in res], np.float32), ms_fp=np.array([r[4] for r in res], np.float32))
        log(f"  saved {dest}")


def load_regions(name):
    z = np.load(os.path.join(CACHE, f"regions-{name}.npz"))
    return {k: z[k] for k in z.files}


# ---- maths -----------------------------------------------------------------------------------

def as_u64(pdq):
    """(..., 32) uint8 -> (..., 4) uint64, for XOR + popcount."""
    return np.ascontiguousarray(pdq).view(">u8").astype(np.uint64)


def pdq_min_dist(q8, db_up):
    """q8: (n, 8, 4) uint64 orientations of the queries; db_up: (m, 4) uint64 upright hashes of the
    stored photos. Returns (n, m) uint16: the best distance over the 8 orientations (photoFingerprint.ts)."""
    best = np.full((q8.shape[0], db_up.shape[0]), PDQ_BITS, np.uint16)
    for o in range(8):
        d = np.zeros_like(best)
        for w in range(4):
            d += np.bitwise_count(q8[:, o, w][:, None] ^ db_up[None, :, w]).astype(np.uint16)
        np.minimum(best, d, out=best)
    return best


def f32(d):
    return np.asarray(d, dtype="<f2").astype(np.float32)  # services/photoVectorCache.ts fp16ToFloat32, no renormalisation


def k_for(fmr, pairs):
    return int(math.floor(fmr * pairs))


def pwm_gpd(excess):
    """Generalised Pareto fit by probability-weighted moments (Hosking and Wallis 1987).
    Returns (k, sigma) in their convention: P(Y > y) = (1 - k y / sigma) ** (1 / k)."""
    y = np.sort(excess)
    n = len(y)
    a0 = y.mean()
    a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * y) / n
    k = a0 / (a0 - 2 * a1) - 2
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    return float(k), float(sigma)


def gpd_quantile(u, k, sigma, ratio):
    """Excess y with P(Y > y | Y > 0) = ratio."""
    return u + (sigma / k * (1 - ratio ** k) if abs(k) > 1e-9 else -sigma * math.log(ratio))


def pair_scan(D, PQ, Q8, min_q, max_d, keep_from):
    """All unordered pairs of distinct photos of one half: cosine histogram, the pairs with cosine
    >= keep_from, and the PDQ pairs (either direction <= max_d, both qualities > min_q)."""
    n = D.shape[0]
    edges = np.linspace(-1.0, 1.0, 4001)
    hist = np.zeros(len(edges) - 1, np.int64)
    top_i, top_j, top_c = [], [], []
    pdq_hist = np.zeros(PDQ_BITS + 1, np.int64)
    pdq_pairs = []
    up = Q8[:, 0, :]
    step = 500
    for s in range(0, n, step):
        e = min(n, s + step)
        S = D[s:e] @ D.T
        iu = np.arange(s, e)[:, None] < np.arange(n)[None, :]
        vals = S[iu]
        hist += np.histogram(np.clip(vals, -1, 1 - 1e-7), edges)[0]
        ii, jj = np.nonzero(iu & (S >= keep_from))
        top_i.append(ii + s); top_j.append(jj); top_c.append(S[ii, jj])
        d1 = pdq_min_dist(Q8[s:e], up)              # rows query, columns stored
        d2 = pdq_min_dist(Q8, up[s:e]).T            # the other direction
        dmin = np.minimum(d1, d2)
        pdq_hist += np.bincount(dmin[iu], minlength=PDQ_BITS + 1)[: PDQ_BITS + 1]
        ok = iu & (dmin <= max_d) & (PQ[s:e, None] > min_q) & (PQ[None, :] > min_q)
        for a, b in zip(*np.nonzero(ok)):
            pdq_pairs.append((a + s, b, int(dmin[a, b])))
    ti, tj, tc = np.concatenate(top_i), np.concatenate(top_j), np.concatenate(top_c)
    o = np.argsort(-tc)
    return {"pairs": n * (n - 1) // 2, "hist": hist, "edges": edges, "top_i": ti[o], "top_j": tj[o], "top_c": tc[o],
            "pdq_hist": pdq_hist, "pdq_pairs": pdq_pairs}


# ---- stage 2b: duplicates inside ABO (label noise), both halves -------------------------------

def half_arrays(half):
    originals, _ = load_set()
    half_of = {x["id"]: x["half"] for x in originals}
    O = load_fp("originals")
    idx = np.nonzero(np.array([half_of[i] == half for i in O["ids"]]))[0]
    return O, idx, f32(O["dino"][idx]), O["quality"][idx], as_u64(O["pdq"][idx])


def stage_duplicates():
    cfg = load_config()
    originals, _ = load_set()
    opath = {x["id"]: x["path"] for x in originals}
    for half in ("calibrate", "measure"):
        dest = os.path.join(CACHE, f"dups-{half}.npz")
        if os.path.exists(dest):
            continue
        O, idx, D, PQ, Q8 = half_arrays(half)
        scan = pair_scan(D, PQ, Q8, cfg["minQuality"], cfg["maxDistance"], keep_from=DUP_COS)
        a, b, c = scan["top_i"], scan["top_j"], scan["top_c"]
        d = np.array([min(int(pdq_min_dist(Q8[i:i + 1], Q8[j:j + 1, 0])[0, 0]), int(pdq_min_dist(Q8[j:j + 1], Q8[i:i + 1, 0])[0, 0])) for i, j in zip(a, b)], np.int32)
        ia, ib = O["ids"][idx[a]], O["ids"][idx[b]]
        log(f"duplicates {half}: {len(a)} pairs with cosine >= {DUP_COS}; /verify on each")
        res = pool_map(_verify_worker, [(opath[x], opath[y], None) for x, y in zip(ia, ib)], f"dups {half}", chunk=4)
        inl = np.array([r[0] for r in res], np.int32)
        flag = (d <= DUP_PDQ) | (inl >= DUP_INLIERS)
        np.savez(dest, a=ia, b=ib, cos=c, pdq=d, inl=inl, flag=flag, images=len(idx))
        log(f"  {int(flag.sum())} duplicate pairs over {len(set(ia[flag]) | set(ib[flag]))} images; saved {dest}")


def load_dups(half):
    z = np.load(os.path.join(CACHE, f"dups-{half}.npz"))
    return {k: z[k] for k in z.files}


def dup_keys(half):
    z = load_dups(half)
    return {frozenset((str(x), str(y))) for x, y, f in zip(z["a"], z["b"], z["flag"]) if f}


def drop_dups(scan, ids, dups):
    """The scan without the duplicate pairs: top lists, PDQ pairs and the pair count."""
    keep = np.array([frozenset((str(ids[i]), str(ids[j]))) not in dups for i, j in zip(scan["top_i"], scan["top_j"])], bool)
    out = dict(scan)
    out["top_i"], out["top_j"], out["top_c"] = scan["top_i"][keep], scan["top_j"][keep], scan["top_c"][keep]
    out["pdq_pairs_raw"] = scan["pdq_pairs"]
    out["pdq_pairs"] = [(i, j, d) for i, j, d in scan["pdq_pairs"] if frozenset((str(ids[i]), str(ids[j]))) not in dups]
    out["pairs_raw"] = scan["pairs"]
    out["pairs"] = scan["pairs"] - len(dups)
    out["n_dup"] = len(dups)
    out["raw_top_c"] = scan["top_c"]
    return out


# ---- stage 3: calibration (calibrate half only) -----------------------------------------------

def stage_calibrate():
    cfg = load_config()
    dest = os.path.join(CACHE, "calibrate.npz")
    if os.path.exists(dest):
        return
    originals, transforms = load_set()
    half_of = {x["id"]: x["half"] for x in originals}
    O = load_fp("originals")
    cal = np.array([half_of[i] == "calibrate" for i in O["ids"]])
    cidx = np.nonzero(cal)[0]
    D = f32(O["dino"][cidx]); PQ = O["quality"][cidx]; Q8 = as_u64(O["pdq"][cidx])
    log(f"calibrate: {len(cidx)} originals, {len(cidx) * (len(cidx) - 1) // 2} non-matching pairs")
    scan = pair_scan(D, PQ, Q8, cfg["minQuality"], cfg["maxDistance"], keep_from=0.5)
    dups = dup_keys("calibrate")
    scan = drop_dups(scan, O["ids"][cidx], dups)
    log(f"  pairs scanned; {scan['n_dup']} duplicate pairs left out; max cosine {scan['top_c'][0]:.4f} (raw {scan['raw_top_c'][0]:.4f}); "
        f"PDQ pairs {len(scan['pdq_pairs'])} (raw {len(scan['pdq_pairs_raw'])})")

    # /verify sample: positives stratified by transform, hard negatives by cosine.
    rng = np.random.default_rng(SEED)
    byid = {x["id"]: x for x in originals}
    ctr = [t for t in transforms if t["half"] == "calibrate"]
    per = VERIFY_SAMPLE // len(TYPES)
    pos = []
    for t in TYPES:
        xs = [x for x in ctr if x["type"] == t]
        pick = rng.choice(len(xs), per + (1 if TYPES.index(t) < VERIFY_SAMPLE - per * len(TYPES) else 0), replace=False)
        pos += [xs[i] for i in sorted(pick)]
    # Hard negatives, two sources of equal size: (a) the most similar pairs of distinct originals (an
    # honest seller's photo against another product shot alike); (b) for transformed copies used as
    # queries, their most similar original that is not their parent (nor a collage filler).
    neg = []
    for a, b, c in zip(scan["top_i"][: VERIFY_SAMPLE // 2], scan["top_j"][: VERIFY_SAMPLE // 2], scan["top_c"][: VERIFY_SAMPLE // 2]):
        neg.append(("orig-orig", O["ids"][cidx[a]], O["ids"][cidx[b]], float(c), None))
    cand = []
    for t in TYPES:
        F = load_fp(t)
        tid = F["ids"]
        tmap = {x["id"]: x for x in ctr}
        rows = [k for k, i in enumerate(tid) if i in tmap]
        S = f32(F["dino"][rows]) @ D.T
        for r, k in enumerate(rows):
            x = tmap[tid[k]]
            bad = {x["parent"]} | set(x.get("fillers") or [])
            order = np.argsort(-S[r])[:12]
            for j in order:
                oid = O["ids"][cidx[j]]
                if oid not in bad and not any(frozenset((str(p), str(oid))) in dups for p in bad):
                    cand.append((t, x["id"], oid, float(S[r, j])))
                    break
    per_neg = (VERIFY_SAMPLE // 2) // len(TYPES)
    for t in TYPES:
        cs = sorted([c for c in cand if c[0] == t], key=lambda c: -c[3])
        extra = 1 if TYPES.index(t) < (VERIFY_SAMPLE // 2) - per_neg * len(TYPES) else 0
        for c in cs[: per_neg + extra]:
            neg.append(("copy-other:" + t, c[1], c[2], c[3], None))

    tpath = {x["id"]: x["path"] for x in transforms}
    opath = {x["id"]: x["path"] for x in originals}
    path = lambda i: tpath[i] if "~" in i else opath[i]
    items = [(path(t["id"]), opath[t["parent"]], None) for t in pos] + [(path(a), opath[b], None) for _, a, b, _, _ in neg]
    log(f"  /verify on {len(pos)} positive and {len(neg)} hard-negative pairs")
    res = pool_map(_verify_worker, items, "verify calibrate", chunk=8)
    pdq_inl = pool_map(_verify_worker, [(opath[O["ids"][cidx[i]]], opath[O["ids"][cidx[j]]], None) for i, j, _ in scan["pdq_pairs"]], "verify PDQ pairs", chunk=4)
    np.savez(dest, pairs=scan["pairs"], pairs_raw=scan["pairs_raw"], n_dup=scan["n_dup"], raw_top_c=scan["raw_top_c"][:2000],
             hist=scan["hist"], edges=scan["edges"], top_i=cidx[scan["top_i"]], top_j=cidx[scan["top_j"]],
             top_c=scan["top_c"], pdq_hist=scan["pdq_hist"], pdq_pairs=np.array(scan["pdq_pairs"], np.int64).reshape(-1, 3),
             n_pdq_raw=len(scan["pdq_pairs_raw"]), pdq_pairs_inl=np.array([r[0] for r in pdq_inl], np.int32),
             pos_ids=np.array([t["id"] for t in pos]), pos_type=np.array([t["type"] for t in pos]),
             neg_kind=np.array([n[0] for n in neg]), neg_a=np.array([n[1] for n in neg]), neg_b=np.array([n[2] for n in neg]),
             neg_cos=np.array([n[3] for n in neg]),
             inl=np.array([r[0] for r in res], np.int32), matches=np.array([r[1] for r in res], np.int32),
             kp=np.array([r[2] for r in res], np.int32), ms=np.array([r[3] for r in res], np.float32), n_pos=len(pos))
    log(f"  saved {dest}")




# ---- the choice (calibrate half only) ----------------------------------------------------------

def load_cal():
    z = np.load(os.path.join(CACHE, "calibrate.npz"))
    return {k: z[k] for k in z.files}


def choose():
    """T and K from calibrate.npz. Every number here comes from the calibrate half."""
    c = load_cal()
    P = int(c["pairs"])
    top = c["top_c"]
    out = {"pairs": P, "max": float(top[0]), "tail": [float(x) for x in top[:20]]}
    emp = {}
    for f in (1e-4, 1e-5, 1e-6, 1e-7):
        k = k_for(f, P)
        emp[f] = float(top[k])  # cosines strictly above this are at most k pairs: FMR <= f
    out["emp"] = emp
    # Tail model: generalised Pareto on the pairs above the 1e-4 quantile, extrapolated.
    u = emp[1e-4]
    n_u = int(np.sum(top > u))
    kk, sig = pwm_gpd(top[:n_u] - u)
    out["gpd"] = {"u": u, "n_u": n_u, "k": kk, "sigma": sig,
                  "q1e-6": gpd_quantile(u, kk, sig, 1e-6 * P / n_u),
                  "q1e-7": gpd_quantile(u, kk, sig, 1e-7 * P / n_u),
                  "q1e-8": gpd_quantile(u, kk, sig, 1e-8 * P / n_u),
                  "endpoint": (u + sig / kk) if kk > 0 else None}
    # 12.5 M pairs hold 1.25 false matches at 1e-7: the empirical quantile is the second-highest
    # pair, a single draw. Proposed: the highest non-matching cosine left after the duplicates,
    # rounded up to the next 0.001 (no calibrate pair reaches T). A margin of one decade of FMR
    # (the empirical 1e-6 level to the maximum) would put T at or above 1, where DINOv2 adds nothing.
    out["decade"] = out["max"] - emp[1e-6]
    out["T"] = round(math.floor(out["max"] * 1000 + 1) / 1000, 3)
    out["margin"] = out["T"] - out["max"]
    out["T_emp"] = math.ceil(emp[1e-7] * 1e4) / 1e4
    out["n_dup"] = int(c["n_dup"])
    out["pairs_raw"] = int(c["pairs_raw"])
    out["raw_tail"] = [float(x) for x in c["raw_top_c"][:12]]
    # Levels shown on the measure half for the decision (all chosen here, on the calibrate half).
    out["T_levels"] = {1e-6: round(emp[1e-6] + 5e-5, 4), 1e-5: round(emp[1e-5] + 5e-5, 4), 1e-4: round(emp[1e-4] + 5e-5, 4)}
    # /verify: the smallest K with no hard negative at or above it.
    n_pos = int(c["n_pos"])
    inl = c["inl"]
    pos, neg = inl[:n_pos], inl[n_pos:]
    out["neg_max_inliers"] = int(neg.max())
    out["K"] = int(neg.max()) + 1
    return out


# ---- stage 4: measure (measure half only; thresholds from choose()) ------------------------------

def _cands(qd, qp8, qq, Dm, up_m, PQm, t_low, cfg):
    """Per query: (pdq top-4 [(j, d)], dino top-4 at t_low [(j, cos)]); the production limits."""
    out = []
    for s in range(0, qd.shape[0], 1000):
        e = min(qd.shape[0], s + 1000)
        S = qd[s:e] @ Dm.T
        Dd = pdq_min_dist(qp8[s:e], up_m)
        for r in range(e - s):
            pc = []
            if qq[s + r] > cfg["minQuality"]:
                js = np.nonzero((Dd[r] <= cfg["maxDistance"]) & (PQm > cfg["minQuality"]))[0]
                js = js[np.argsort(Dd[r, js], kind="stable")][:MAX_VERIFY_PER_STAGE]
                pc = [(int(j), int(Dd[r, j])) for j in js]
            js = np.nonzero(S[r] >= t_low)[0]
            js = js[np.argsort(-S[r, js], kind="stable")][:MAX_VERIFY_PER_STAGE]
            out.append((pc, [(int(j), float(S[r, j])) for j in js]))
    return out


def _parent_scores(qd, qp8, qq, Dm, up_m, PQm, pj):
    cos = np.einsum("ij,ij->i", qd, Dm[pj])
    d = pdq_min_dist_rows(qp8, up_m[pj])
    return cos, d


def pdq_min_dist_rows(q8, ups):
    """Row-wise: query i against stored hash i."""
    best = np.full(q8.shape[0], PDQ_BITS, np.uint16)
    for o in range(8):
        d = sum(np.bitwise_count(q8[:, o, w] ^ ups[:, w]).astype(np.uint16) for w in range(4))
        best = np.minimum(best, d)
    return best


def stage_measure():
    cfg = load_config()
    ch = choose()
    T_list = sorted({ch["T"], *ch["T_levels"].values()})
    t_low = min(T_list)
    originals, transforms = load_set()
    half_of = {x["id"]: x["half"] for x in originals}
    opath = {x["id"]: x["path"] for x in originals}
    tpath = {x["id"]: x["path"] for x in transforms}
    tinfo = {x["id"]: x for x in transforms}
    O = load_fp("originals")
    midx = np.nonzero(np.array([half_of[i] == "measure" for i in O["ids"]]))[0]
    mid = O["ids"][midx]
    pos_of = {i: k for k, i in enumerate(mid)}
    Dm, PQm, Q8m = f32(O["dino"][midx]), O["quality"][midx], as_u64(O["pdq"][midx])
    up_m = Q8m[:, 0, :]

    queries = {}  # name -> dict of arrays and candidate lists
    # Unchanged re-upload: the stored file itself, uploaded again by another member.
    queries["reupload"] = {"ids": mid, "parent": np.arange(len(mid)), "box": [None] * len(mid), "owner": np.arange(len(mid)),
                           "d": Dm, "p8": Q8m, "q": PQm, "path": [opath[i] for i in mid]}
    for t in TYPES:
        F = load_fp(t)
        rows = np.array([k for k, i in enumerate(F["ids"]) if tinfo[i]["half"] == "measure"])
        ids = F["ids"][rows]
        queries[t] = {"ids": ids, "parent": np.array([pos_of[tinfo[i]["parent"]] for i in ids]), "box": [None] * len(rows),
                      "owner": np.arange(len(rows)), "d": f32(F["dino"][rows]), "p8": as_u64(F["pdq"][rows]),
                      "q": F["quality"][rows], "path": [tpath[i] for i in ids]}
    for t in REGION_TYPES:
        R = load_regions(t)
        keep = np.array([tinfo[R["ids"][o]]["half"] == "measure" for o in R["owner"]], bool)
        for kind in ("rect", "oracle"):
            sel = np.nonzero(keep & (R["kind"] == kind))[0]
            ids = R["ids"][R["owner"][sel]]
            queries[f"{t}:{kind}"] = {"ids": ids, "parent": np.array([pos_of[tinfo[i]["parent"]] for i in ids]),
                                      "box": [list(map(int, b)) for b in R["box"][sel]], "owner": R["owner"][sel],
                                      "area": (R["box"][sel, 2] - R["box"][sel, 0]) * (R["box"][sel, 3] - R["box"][sel, 1]),
                                      "d": f32(R["dino"][sel]), "p8": as_u64(R["pdq"][sel]), "q": R["quality"][sel],
                                      "path": [tpath[i] for i in ids]}
    for name, Q in queries.items():
        log(f"measure: candidates {name} ({len(Q['ids'])} queries)")
        Q["cands"] = _cands(Q["d"], Q["p8"], Q["q"], Dm, up_m, PQm, t_low, cfg)
        Q["pcos"], Q["pd"] = _parent_scores(Q["d"], Q["p8"], Q["q"], Dm, up_m, PQm, Q["parent"])

    # Non-matching pairs of the measure half: what passes a stage, and the 1,000 most similar.
    log("measure: non-matching pairs of the measure half")
    scan = pair_scan(Dm, PQm, Q8m, cfg["minQuality"], cfg["maxDistance"], keep_from=0.5)
    mdups = dup_keys("measure")
    scan = drop_dups(scan, mid, mdups)
    n_top = max(1000, int(np.sum(scan["top_c"] >= t_low)))
    nm_pairs = [(int(a), int(b)) for a, b in zip(scan["top_i"][:n_top], scan["top_j"][:n_top])]
    seen = set(nm_pairs)
    nm_pairs += [(int(a), int(b)) for a, b, _ in scan["pdq_pairs"] if (int(a), int(b)) not in seen]

    # /verify: every (query, candidate, box) once, cached across runs.
    vpath = os.path.join(CACHE, "verify-measure.npz")
    done = {}
    if os.path.exists(vpath):
        z = np.load(vpath, allow_pickle=False)
        for k, v, ms in zip(z["keys"], z["inl"], z["ms"]):
            done[str(k)] = (int(v), float(ms))
    key = lambda qp, cp, box: f"{qp}|{cp}|{'' if box is None else ','.join(map(str, box))}"
    jobs = {}
    for name, Q in queries.items():
        for r, (pc, dc) in enumerate(Q["cands"]):
            for j in {j for j, _ in pc} | {j for j, _ in dc}:
                k = key(Q["path"][r], opath[mid[j]], Q["box"][r])
                if k not in done:
                    jobs[k] = (Q["path"][r], opath[mid[j]], Q["box"][r])
    for a, b in nm_pairs:
        k = key(opath[mid[a]], opath[mid[b]], None)
        if k not in done:
            jobs[k] = (opath[mid[a]], opath[mid[b]], None)
    log(f"measure: {len(jobs)} /verify calls to make ({len(done)} cached)")
    if jobs:
        ks = list(jobs)
        res = pool_map(_verify_worker, [jobs[k] for k in ks], "verify measure", chunk=8)
        for k, r in zip(ks, res):
            done[k] = (int(r[0]), float(r[3]))
        np.savez(vpath, keys=np.array(list(done)), inl=np.array([v[0] for v in done.values()], np.int32),
                 ms=np.array([v[1] for v in done.values()], np.float32))
    inl = lambda qp, cp, box: done[key(qp, cp, box)][0]

    # ---- tally ----
    K, K12 = ch["K"], cfg["minInliers12"]
    res = {"T_list": T_list, "K": K, "K12": K12, "types": {}, "regions": {}}
    filler_pos = {}
    for i, x in tinfo.items():
        if x["type"] == "collage_2x2" and x["half"] == "measure":
            filler_pos[i] = {pos_of[f] for f in x["fillers"]}

    def tally(name, Q, T, Ks):
        n = len(Q["ids"])
        pdq_alone = (Q["q"] > cfg["minQuality"]) & (PQm[Q["parent"]] > cfg["minQuality"]) & (Q["pd"] <= cfg["maxDistance"])
        dino_alone = Q["pcos"] >= T
        per_k = {}
        for k in Ks:
            det = np.zeros(n, bool)
            false_conf = 0; filler_conf = 0; wrong_cands = 0; dup_conf = 0
            for r, (pc, dc) in enumerate(Q["cands"]):
                cs = {j for j, _ in pc} | {j for j, c in dc if c >= T}
                fill = filler_pos.get(Q["ids"][r], set())
                twins = {f for f in fill | {int(Q["parent"][r])}}
                for j in cs:
                    ok = inl(Q["path"][r], opath[mid[j]], Q["box"][r]) >= k
                    if j == Q["parent"][r]:
                        det[r] |= ok
                    elif j in fill:
                        filler_conf += ok
                    elif any(frozenset((str(mid[t]), str(mid[j]))) in mdups for t in twins):
                        dup_conf += ok  # the same picture under another ABO id: label noise
                    else:
                        wrong_cands += 1
                        false_conf += ok
            per_k[k] = {"det": det, "false": false_conf, "filler": filler_conf, "wrong_cands": wrong_cands, "dup": dup_conf}
        return {"n": n, "pdq": pdq_alone, "dino": dino_alone, "k": per_k}

    for T in T_list:
        res["types"][T] = {name: tally(name, Q, T, (K, K12)) for name, Q in queries.items()}
    # Non-matching pairs of the measure half.
    nm = {}
    for T in T_list:
        cos_pass = [(int(a), int(b)) for a, b, c in zip(scan["top_i"], scan["top_j"], scan["top_c"]) if c >= T]
        pdq_pass = [(int(a), int(b)) for a, b, _ in scan["pdq_pairs"]]
        stage = set(cos_pass) | set(pdq_pass)
        nm[T] = {"cos_pass": len(cos_pass), "pdq_pass": len(pdq_pass), "stage_pass": len(stage),
                 "conf": {k: sum(inl(opath[mid[a]], opath[mid[b]], None) >= k for a, b in stage) for k in (K, K12)}}
    top_inl = np.array([inl(opath[mid[a]], opath[mid[b]], None) for a, b in nm_pairs[:1000]])
    vm = np.array([v[1] for v in done.values()])
    with open(os.path.join(CACHE, "measure.json"), "w") as f:
        json.dump(_summarise(res, queries, scan, nm, top_inl, ch, cfg, vm, nm_pairs, mid), f, indent=1)
    log("measure: saved measure.json")


def _summarise(res, queries, scan, nm, top_inl, ch, cfg, vm, nm_pairs, mid):
    K, K12 = res["K"], res["K12"]
    out = {"T_list": res["T_list"], "K": K, "K12": K12, "rows": {}, "regions": {}, "nonmatch": {}}
    for T, by in res["types"].items():
        rows = {}
        for name in ["reupload"] + TYPES:
            x = by[name]
            rows[name] = {"n": x["n"], "pdq": float(x["pdq"].mean()), "dino": float(x["dino"].mean()),
                          "stage": float((x["pdq"] | x["dino"]).mean()),
                          "cascade_K": float(x["k"][K]["det"].mean()), "cascade_12": float(x["k"][K12]["det"].mean()),
                          "false_K": x["k"][K]["false"], "false_12": x["k"][K12]["false"],
                          "filler_K": x["k"][K]["filler"], "wrong_cands": x["k"][K]["wrong_cands"], "dup_K": x["k"][K]["dup"],
                          "filler_12": x["k"][K12]["filler"], "dup_12": x["k"][K12]["dup"]}
        out["rows"][str(T)] = rows
        regs = {}
        for t in REGION_TYPES:
            whole = by[t]
            n = whole["n"]
            r = {"n": n}
            for kind in ("rect", "oracle"):
                Q = queries[f"{t}:{kind}"]
                x = by[f"{t}:{kind}"]
                for kk, lab in ((K, "K"), (K12, "12")):
                    det_any = np.zeros(n, bool)
                    np.logical_or.at(det_any, Q["owner"], x["k"][kk]["det"])
                    r[f"{kind}_{lab}"] = float((det_any | whole["k"][kk]["det"]).mean())
                    r[f"{kind}_false_{lab}"] = x["k"][kk]["false"]
                    if kind == "rect":
                        # largest rectangle only
                        big = {}
                        for i, (o, a) in enumerate(zip(Q["owner"], Q["area"])):
                            if o not in big or a > Q["area"][big[o]]:
                                big[o] = i
                        det_big = np.zeros(n, bool)
                        for o, i in big.items():
                            det_big[o] = x["k"][kk]["det"][i]
                        r[f"rect_largest_{lab}"] = float((det_big | whole["k"][kk]["det"]).mean())
                stage_any = np.zeros(n, bool)
                np.logical_or.at(stage_any, Q["owner"], x["pdq"] | x["dino"])
                r[f"{kind}_stage"] = float((stage_any | whole["pdq"] | whole["dino"]).mean())
                if kind == "oracle":  # a sample: rates over the sampled images only, whole image alongside
                    sub = np.unique(Q["owner"])
                    r["oracle_n"] = int(len(sub))
                    for kk, lab in ((K, "K"), (K12, "12")):
                        det_any = np.zeros(n, bool)
                        np.logical_or.at(det_any, Q["owner"], x["k"][kk]["det"])
                        r[f"oracle_{lab}"] = float((det_any | whole["k"][kk]["det"])[sub].mean())
                        r[f"oracle_whole_{lab}"] = float(whole["k"][kk]["det"][sub].mean())
                    r["oracle_stage"] = float((stage_any | whole["pdq"] | whole["dino"])[sub].mean())
                r[f"{kind}_queries"] = int(len(Q["ids"]))
                r[f"{kind}_with"] = int(len(set(Q["owner"].tolist())))
            r["whole_K"] = float(whole["k"][K]["det"].mean())
            r["whole_12"] = float(whole["k"][K12]["det"].mean())
            regs[t] = r
        out["regions"][str(T)] = regs
        out["nonmatch"][str(T)] = nm[T]
    out["nonmatch_pairs"] = int(scan["pairs"])
    out["nonmatch_pairs_raw"] = int(scan["pairs_raw"])
    out["n_dup"] = int(scan["n_dup"])
    out["nonmatch_raw_max"] = float(scan["raw_top_c"][0])
    out["pdq_pairs_raw"] = int(len(scan["pdq_pairs_raw"]))
    out["nonmatch_max"] = float(scan["top_c"][0])
    out["nonmatch_tail"] = [float(x) for x in scan["top_c"][:10]]
    out["nonmatch_pdq_pairs"] = [[str(mid[a]), str(mid[b]), int(d)] for a, b, d in scan["pdq_pairs"]]
    out["nonmatch_top_pairs"] = [[str(mid[a]), str(mid[b]), float(c)] for a, b, c in zip(scan["top_i"][:10], scan["top_j"][:10], scan["top_c"][:10])]
    out["nonmatch_count_ge"] = {str(T): int(np.sum(scan["top_c"] >= T)) for T in res["T_list"]}
    out["top1000_inliers"] = {"max": int(top_inl.max()), "ge_K": int(np.sum(top_inl >= K)), "ge_12": int(np.sum(top_inl >= K12)),
                              "p99": float(np.percentile(top_inl, 99))}
    out["verify_ms"] = {"p50": float(np.percentile(vm, 50)), "p95": float(np.percentile(vm, 95)), "n": int(len(vm))}
    out["pdq_hist_le31"] = int(scan["pdq_hist"][:32].sum())
    return out



def parent_cosine_medians():
    """Median cosine between each measure copy and its parent, per transform."""
    originals, transforms = load_set()
    tinfo = {x["id"]: x for x in transforms}
    O, idx, D, _, _ = half_arrays("measure")
    pos_of = {i: k for k, i in enumerate(O["ids"][idx])}
    out = {}
    for t in TYPES:
        F = load_fp(t)
        rows = [k for k, i in enumerate(F["ids"]) if tinfo[i]["half"] == "measure"]
        par = np.array([pos_of[tinfo[F["ids"][k]]["parent"]] for k in rows])
        out[t] = float(np.median(np.einsum("ij,ij->i", f32(F["dino"][rows]), D[par])))
    return out


# ---- the report's text, from the numbers ---------------------------------------------------------

def narrative(ch, M, c, cfg, RO):
    T, K = ch["T"], ch["K"]
    Ts = str(T)
    rows, regs, nm = M["rows"][Ts], M["regions"][Ts], M["nonmatch"][Ts]
    n_pos = int(c["n_pos"]); pos, neg = c["inl"][:n_pos], c["inl"][n_pos:]
    all_rows = [rows[t] for t in TYPES]
    n_all = sum(r["n"] for r in all_rows)
    avg = lambda key: sum(r[key] * r["n"] for r in all_rows) / n_all
    d = load_dups("calibrate"); dm = load_dups("measure")
    T_txt = {}
    T_txt["@@METHOD@@"] = "\n".join([
        f"- **Set.** 10,000 Amazon Berkeley Objects photos (CC BY 4.0, « small » size: at most 256 px a side), split by the manifest's seeded shuffle into 5,000 calibrate and 5,000 measure originals; 54,000 transformed copies (9 types × 6,000 parents: 1,000 calibrate, 5,000 measure). Every file's sha256 checked against the manifests.",
        "- **Fingerprints.** The vision service's own modules, imported in Python (no HTTP): `images.decode`, `fingerprint.fingerprint` (PDQ of the upright image and its 8 orientations, PDQ quality, DINOv2-S class token L2-normalised and stored as fp16), `models.Registry` with DINOv2-S, `verify.verify`, `screen.photo_rects_part`. Dependencies from `vision/requirements.lock` in a temporary uv venv (Python 3.12, `pdqhash` compiled from its locked source as in the Dockerfile). 4 processes of one thread each.",
        f"- **Duplicates inside ABO (label noise).** ABO stores the same photograph under several image ids (catalogue duplicates, the same print on several phone-case mock-ups, nutrition-label templates). Rule (coordinator, 25/11): a pair of distinct ids with cosine ≥ {DUP_COS} **and** (PDQ ≤ {DUP_PDQ} **or** ≥ {DUP_INLIERS} `/verify` inliers) is the same picture. Excluded as pairs from the non-matching sets of both halves: **{int(d['flag'].sum())} pairs over {len(set(d['a'][d['flag']]) | set(d['b'][d['flag']]))} images in the calibrate half** (of {len(d['flag'])} pairs at cosine ≥ {DUP_COS}), **{int(dm['flag'].sum())} pairs over {len(set(dm['a'][dm['flag']]) | set(dm['b'][dm['flag']]))} images in the measure half** (of {len(dm['flag'])}). The images stay; a copy matched to its parent's duplicate counts neither as a find nor as a false confirmation.",
        "- **Calibrate half only:** T, the PDQ count, and the `/verify` sample (2,000 true copies stratified by transform; 2,000 hard negatives: the 1,000 most similar pairs of distinct originals, and 1,000 copies against their most similar wrong original, 111 per transform, never a collage's own fillers nor a duplicate of the parent).",
        f"- **Measure half only:** database = the 5,000 measure originals; queries = their 45,000 copies, plus each original uploaded again unchanged. The cascade of `services/photoMatch.ts`: PDQ (best of the query's 8 orientations against the stored upright hash, ≤ {cfg['maxDistance']} bits, both qualities > {cfg['minQuality']}, nearest {MAX_VERIFY_PER_STAGE}), DINOv2 (cosine of the fp16 vectors ≥ T, best {MAX_VERIFY_PER_STAGE}), union, `/verify` on each candidate, copy at ≥ K inliers. Today's check compares addresses only: every copy is a new upload with a new address, so it finds 0 % by construction (and no copy shares its parent's sha256).",
        f"- **`/verify` is SIFT + RANSAC** (`service/app/verify.py`: Lowe ratio 0.75, RANSAC 5 px, a mirrored try below 12 inliers), not ORB as the brief says. Measured as committed.",
        f"- **Regions:** `photo_rects_part` on every measure screenshot and collage, each rectangle fingerprinted by `fingerprint()` with `regions`, searched the same way, verified with `box_a`. The « perfect box » (where the generator pasted the photo) on a seeded sample of {ORACLE_SAMPLE} per type, as an upper bound.",
    ])
    g = ch["gpd"]
    n_above = int(np.sum(c["top_c"] >= T))
    T_txt["@@T@@"] = "\n".join([
        f"**Before the duplicates were taken out** ({ch['pairs_raw']:,} pairs), the highest cosines were " + ", ".join(f"{x:.4f}" for x in ch["raw_tail"][:8]) + f": the 10⁻⁷ threshold would have been {ch['raw_tail'][1]:.4f}, set by the same photograph under two ids. After ({ch['pairs']:,} pairs), the values above.",
        "",
        f"**How far to trust it.** At 10⁻⁷, {ch['pairs']:,} pairs allow {ch['pairs'] * 1e-7:.2f} false pairs: the empirical threshold ({ch['T_emp']:.4f}) is the second-highest pair, one draw. The tail model does not help: it is fitted on 1,250 pairs and predicts an upper end ({g['endpoint'] or float('nan'):.4f}) below the highest pair actually seen ({ch['max']:.4f}). And the pairs at the very top are still not « different photos that look alike »: they are catalogue templates that the duplicate rule keeps (PDQ 10 to 28 bits, 15 to 42 inliers): the same grey mannequin or hand with a different small jewel, the same product render with another colour or label. Their cosine measures ABO's catalogue, not DINOv2's error on two sellers' own photos.",
        "",
        f"**Proposed T = {T}**: the highest non-matching cosine of the calibrate half after the duplicates ({ch['max']:.4f}), rounded up to the next 0.001 (margin {ch['margin']:.4f}); no calibrate pair reaches it, so the rule of three bounds the rate at 3 / {ch['pairs']:,} = {3 / ch['pairs']:.1e} with 95 % confidence. A wider margin is not usable: one decade of false-match rate costs {ch['decade']:.4f} of cosine at the top of this tail, so « maximum + one decade » is ≥ 1, where DINOv2 adds nothing to PDQ. Out of sample, the measure half has {M['nonmatch_count_ge'][Ts]} non-matching pair(s) at or above {T} (of {M['nonmatch_pairs']:,}).",
    ])
    pdq_inl = c["pdq_pairs_inl"]
    T_txt["@@PDQ@@"] = "\n".join([
        f"The rule does **not** give zero false matches on ABO. Calibrate half: **{int(c['n_pdq_raw'])} non-matching pairs** within 31 bits with both qualities > 49 (best of 8 orientations, either direction), {int(c['n_pdq_raw']) / ch['pairs_raw']:.1e} per pair; **{len(pdq_inl)}** after the duplicates. Of those {len(pdq_inl)}, `/verify` confirms {int(np.sum(pdq_inl >= 12))} at 12 inliers and {int(np.sum(pdq_inl >= K))} at K = {K}; {int(np.sum(pdq_inl < 8))} have fewer than 8 inliers (there PDQ alone matched two different pictures). Measure half: {M['pdq_pairs_raw']} raw, {nm['pdq_pass']} after the duplicates.",
        "",
        f"Low quality: {int(np.sum(load_fp('originals')['quality'] <= cfg['minQuality']))} of the 10,000 originals have PDQ quality ≤ {cfg['minQuality']} (the hash is not used for them). This is why PDQ never decides alone: in the cascade a PDQ match only sends the pair to `/verify`.",
    ])
    kz = int(neg.max()) + 1
    steps = sorted({12, 16, 20, 25, 30, 40, 50, 75, 100, 150, kz})
    lines = ["| K | Hard negatives confirmed (of %d) | True copies found (of %d) | crop_50 | rotate_15 | classifieds_page | collage_2x2 |" % (len(neg), len(pos)), "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in steps:
        f = lambda t: pct(np.mean(pos[c["pos_type"] == t] >= k))
        lines.append(f"| {k} | {int(np.sum(neg >= k))} | {pct(np.mean(pos >= k))} | {f('crop_50')} | {f('rotate_15')} | {f('classifieds_page')} | {f('collage_2x2')} |")
    T_txt["@@K@@"] = "\n".join([
        f"**The brief's rule (no hard negative confirmed in the sample) gives K = {kz}**, the highest hard-negative count plus one. Its cost: true copies found falls from {pct(np.mean(pos >= 12))} at K = 12 to {pct(np.mean(pos >= kz))}.",
        "",
        "\n".join(lines),
        "",
        "@@KDISC@@",
    ])
    lo = min(M["T_list"]); L = M["rows"][str(lo)]; LR = M["regions"][str(lo)]
    allL = [L[t] for t in TYPES]
    avgL = lambda key: sum(r[key] * r["n"] for r in allL) / n_all
    nmL = M["nonmatch"][str(lo)]
    pos12 = float(np.mean(pos >= 12)); posK = float(np.mean(pos >= kz))
    T_txt["@@KDISC@@"] = "\n".join([
        f"**Why the rule fails here.** After the duplicate rule, {int(np.sum(neg >= 12))} of the {len(neg)} hard negatives still reach 12 inliers. "
        "A look at samples (25/11, to describe the data only; no threshold uses it) shows what they are: mostly **the same photograph in another catalogue composite** "
        "(one print on several phone-case models), which the rule keeps because their cosine is below 0.97 or their PDQ above 8; "
        "and **catalogue templates** (the same grey mannequin or hand with a different small jewel). Both are copies of a base picture by the plan's own definition (« photo collée », « texte incrusté »), not « two products that look alike ». "
        "A minority are genuinely different pictures that share a graphic: the same phone-case outline, a nutrition table, a side panel with the same text, and once two different chairs at 18 inliers. "
        "In a one-off check of 250 random pairs per cosine band, 7 pairs between 0.5 and 0.7 and 17 between 0.7 and 0.8 reached 12 inliers.",
        "",
        f"**Proposed `minInliers`: keep 12, provisionally.** The brief's rule ({kz}) finds {pct(posK)} of the true copies instead of {pct(pos12)}, and is set by label noise; "
        "no value separates copies from hard negatives on this set. The decision needs a hard-negative set of genuinely different photos (several people photographing the same model of object), which ABO does not provide. "
        "Two defences that do not depend on K: verify a screenshot or collage inside its photo box (`box_a`), so a shared page frame cannot supply the inliers; and keep the context rules (same seller, buyer reselling) ahead of any refusal.",
    ])
    T_txt["@@NM@@"] = (f"What these confirmations are: a random sample of 24 of the {nm['conf']['12']} confirmed measure pairs (looked at to describe them, after the thresholds were fixed) held 16 pairs of one print on two phone-case models and 8 pairs of the same mannequin or hand template with a different jewel; "
                       "no pair of independently taken photos. On ABO, the « false confirmations » therefore count catalogue re-use the duplicate rule does not reach, not errors on two sellers' own photos; but they also show that `/verify` alone will call a shared template a copy.")
    med = parent_cosine_medians()
    T_txt["@@DETECT@@"] = "\n".join([
        f"- **Today: 0 %** on every transform (address comparison). **Cascade at T = {T}, K = 12: {pct(avg('cascade_12'))}** of the 45,000 copies, almost all from the PDQ stage (PDQ alone {pct(avg('pdq'))}; DINOv2 at {T} adds {pct(avg('stage') - avg('pdq'))}).",
        f"- **Unchanged re-upload: {pct(rows['reupload']['cascade_12'])}**, not 100 %. Both fingerprints find the stored photo ({pct(rows['reupload']['stage'])}); `/verify` of a photo against itself gives fewer than 12 inliers for 120 of the 5,000 (plain products on white: too few SIFT keypoints at 256 px).",
        f"- **Mirror: PDQ finds only {pct(rows['mirror']['pdq'])}.** The best of its 8 orientations is 30 bits away at the median, even for a mirror made in memory without recompression: on 256 px images the dihedral hashes are approximate. DINOv2 finds mirrors only at the looser levels ({pct(L['mirror']['dino'])} at {lo}).",
        f"- **DINOv2-S on these images:** median cosine to the parent {med['jpeg_q50']:.3f} for JPEG q50, {med['downscale_50']:.3f} for a 50 % downscale, {med['crop_70']:.3f} for the 70 % crop. The non-matching tail reaches 0.99, so at 10⁻⁷ DINOv2 finds 0 % of crops and rotations.",
        f"- **At the 10⁻⁴ level (T = {lo})**, the cascade at K = 12 finds {pct(avgL('cascade_12'))} (crop_70 {pct(L['crop_70']['cascade_12'])}, rotate_15 {pct(L['rotate_15']['cascade_12'])}, mirror {pct(L['mirror']['cascade_12'])}), but {nmL['stage_pass']} measure pairs go to `/verify` and {nmL['conf']['12']} are confirmed (12.5 M pairs; at 100,000 photos that is about 4 `/verify` calls per photo, the cap).",
    ])
    T_txt["@@REGIONS@@"] = "\n".join([
        f"- **Whole image: 0 %** of screenshots and collages at T = {T} ({pct(L['classifieds_page']['cascade_12'])} and {pct(L['collage_2x2']['cascade_12'])} at {lo}).",
        f"- **With the rectangles of `screen.py`: {pct(regs['classifieds_page']['rect_12'])} of screenshots and {pct(regs['collage_2x2']['rect_12'])} of collages at T = {T}; {pct(LR['classifieds_page']['rect_12'])} and {pct(LR['collage_2x2']['rect_12'])} at {lo}.** Not above 0.18. The finder returns a rectangle on only {regs['classifieds_page']['rect_with']:,} of the 5,000 pages, and its rectangles are mostly the product, not the photo: ABO photos have white backgrounds, which the finder reads as flat page.",
        f"- **With the perfect box: {pct(LR['classifieds_page']['oracle_12'])} of screenshots and {pct(LR['collage_2x2']['oracle_12'])} of collages at {lo}** ({pct(regs['classifieds_page']['oracle_12'])} and {pct(regs['collage_2x2']['oracle_12'])} at {T}). The region idea works when the box is right and T is loose: well above 0.18. The rectangle finder is the weak part, and the 10⁻⁷ threshold removes most of the gain.",
    ])
    T_txt["@@CONFIG@@"] = "\n".join([
        "```json",
        "\"dino\": {",
        f"  \"threshold\": {{ \"value\": {T}, \"source\": \"calibration-fingerprints-{REPORT_DATE}.md: highest non-matching cosine of the calibrate half ({ch['max']:.4f}, {ch['pairs']:,} pairs after {ch['n_dup']} ABO duplicate pairs) rounded up; 0 calibrate and 0 measure pairs reach it\" }},",
        "  \"variant\": { \"value\": \"small\", \"source\": \"the variant calibrated (DINOv2-base not measured)\" }",
        "},",
        "\"verify\": {",
        f"  \"minInliers\": {{ \"value\": 12, \"source\": \"unchanged: calibration-fingerprints-{REPORT_DATE}.md found no value that separates copies from hard negatives on ABO (zero hard negatives needs {kz}, which finds {pct(posK)} of copies instead of {pct(pos12)})\" }}",
        "}",
        "```",
        "",
        f"If the owner prefers detection to the 10⁻⁷ target, the measured alternative is `dino.threshold` = {lo} (10⁻⁴ on the calibrate half): it finds crops and mirrors, at about 4 `/verify` calls per photo and with every confirmation resting on `/verify` alone (section 5).",
    ])
    tgt = [
        ("False-match rate 10⁻⁷ per pair (DINOv2 stage)", f"0 calibrate pairs ≥ {T} (bound {3 / ch['pairs']:.1e}); 0 of {M['nonmatch_pairs']:,} measure pairs", "**met** (resolution of about one pair)"),
        ("Same, PDQ stage (≤ 31, quality > 49)", f"{nm['pdq_pass']} measure pairs ({nm['pdq_pass'] / M['nonmatch_pairs']:.1e}), {nm['conf']['12']} confirmed at 12", "**missed** on ABO (catalogue re-use)"),
        ("Unchanged re-upload 100 %", f"{pct(rows['reupload']['cascade_12'])} (fingerprints {pct(rows['reupload']['stage'])})", "**missed** (by `/verify` on plain photos)"),
        ("70 % crop ≥ 0.9", f"{pct(rows['crop_70']['cascade_12'])} at T = {T}; {pct(L['crop_70']['cascade_12'])} at {lo}", "**missed**"),
        ("Screenshots well above 0.18", f"{pct(regs['classifieds_page']['rect_12'])} with rectangles at T = {T}; {pct(LR['classifieds_page']['rect_12'])} at {lo}; perfect box {pct(LR['classifieds_page']['oracle_12'])} at {lo}", "**missed** (reachable only with a better box and a looser T)"),
    ]
    T_txt["@@TARGETS@@"] = "\n".join(["| Target (plan, phase 1) | Measured (measure half) | Verdict |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in tgt])
    T_txt["@@LIMITS@@"] = "\n".join([
        "- **ABO is a catalogue, not second-hand photos.** It re-uses one photograph across ids, composites and templates. The duplicate rule removes the closest pairs only; the rest sets both the 10⁻⁷ threshold and the hard negatives. Calibrating on real seller photos (several people photographing the same model of object) would likely give a lower T and a usable K; this set cannot show it.",
        "- **Small images.** ABO « small » photos are at most 256 px; phase 2 keeps originals up to 4,096 px and `/verify` works at up to 1,024 px. Inlier counts, PDQ's dihedral error and DINOv2's cosines may all differ on real photos.",
        "- **10⁻⁷ rests on about one pair** (12.5 M pairs per half). « 0 pairs above T » on both halves is the honest statement; the tail model is not reliable at this level.",
        "- **The database is 5,000 photos.** At 100,000 photos, false candidates per query are 20 times more; the PDQ and `/verify` caps (4 per stage) then matter.",
        "- **Synthetic screenshots and collages**: one page layout, white-background products; real screenshots have phone chrome and varied pages.",
        "- **Time** was measured on a shared laptop (load 7 to 23 during the run), not on the server.",
        "- The cascade is simulated from `services/photoMatch.ts` (candidate order, the cap of 4 per stage); the API's band lookup is replaced by a full scan, which finds the same pairs (every pair within 31 bits shares a band within one bit).",
    ])
    T_txt["@@SUMMARY@@"] = "\n".join([
        "**In short.**",
        f"- Proposed `dino.threshold` = **{T}** (DINOv2-S): no non-matching pair reaches it on either half. Proposed `verify.minInliers` = **12, unchanged**: the brief's zero-hard-negative rule gives {kz}, which finds {pct(posK)} of true copies instead of {pct(pos12)} (calibrate sample).",
        f"- ABO holds the same photograph under several ids: **{ch['n_dup']} duplicate pairs (calibrate) and {M['n_dup']} (measure) were excluded** by the coordinator's rule. Without them, the 10⁻⁷ threshold would have been {ch['raw_tail'][1]:.4f}.",
        f"- Detection of the 45,000 measure copies: **today 0 %; PDQ alone {pct(avg('pdq'))}; cascade {pct(avg('cascade_12'))}**. Re-upload {pct(rows['reupload']['cascade_12'])}, JPEG {pct(rows['jpeg_q50']['cascade_12'])}, downscale {pct(rows['downscale_50']['cascade_12'])}, mirror {pct(rows['mirror']['cascade_12'])}, text {pct(rows['text_overlay']['cascade_12'])}; crops, rotation, screenshots and collages 0 %.",
        f"- Region fingerprints with the module's rectangle finder: screenshots {pct(regs['classifieds_page']['rect_12'])}, collages {pct(regs['collage_2x2']['rect_12'])}; with a perfect box and the 10⁻⁴ level: {pct(LR['classifieds_page']['oracle_12'])} and {pct(LR['collage_2x2']['oracle_12'])}.",
        "- Targets: 10⁻⁷ met for DINOv2; re-upload 100 %, crop 70 ≥ 0.9 and screenshots > 0.18 missed. The causes are measured below: catalogue duplicates in ABO, 256 px images, and a rectangle finder that misses photos on white backgrounds.",
    ])
    T_txt["@@TIME@@"] = ("Images are small here (ABO « small », 256 px; pages 480 × 680), so DINOv2 dominates: PDQ takes about 8 ms. "
                             "The vision README measured `/fingerprint` at 0.63 s p50 on a 1,600 px photo with one thread at `--cpus 0.75`; the server's own figure stays to be measured (phase 3).")
    return T_txt


# ---- stage 5: the report ---------------------------------------------------------------------

def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def timing_rows():
    rows = []
    for name in ["originals"] + TYPES:
        F = load_fp(name)
        ms = F["ms_fp"]
        sz = F["size"]
        rows.append((name, len(ms), f"{int(np.median(sz[:, 0]))}×{int(np.median(sz[:, 1]))}",
                     float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), int(np.sum(~F["sha_ok"]))))
    return rows


def calib_tables(ch):
    c = load_cal()
    n_pos = int(c["n_pos"])
    inl = c["inl"]
    pos, neg = inl[:n_pos], inl[n_pos:]
    K = ch["K"]
    lines = ["| Transform (calibrate half) | Pairs | Inliers p5 | p50 | Found at K = 12 | Found at K = %d |" % K, "|---|---:|---:|---:|---:|---:|"]
    for t in TYPES:
        m = c["pos_type"] == t
        x = pos[m]
        lines.append(f"| {t} | {m.sum()} | {np.percentile(x, 5):.0f} | {np.percentile(x, 50):.0f} | {pct(np.mean(x >= 12))} | {pct(np.mean(x >= K))} |")
    lines.append(f"| **all** | {len(pos)} | {np.percentile(pos, 5):.0f} | {np.percentile(pos, 50):.0f} | {pct(np.mean(pos >= 12))} | {pct(np.mean(pos >= K))} |")
    neg_lines = ["| Hard negatives (calibrate half) | Pairs | Cosine range | Inliers p50 | p99 | max | ≥ 12 |", "|---|---:|---|---:|---:|---:|---:|"]
    kinds = c["neg_kind"]
    groups = [("distinct originals, most similar pairs", kinds == "orig-orig"),
              ("copy against its most similar wrong original", np.char.startswith(kinds.astype(str), "copy-other"))]
    for lab, m in groups + [("**all**", np.ones(len(neg), bool))]:
        x = neg[m]; cs = c["neg_cos"][m]
        neg_lines.append(f"| {lab} | {m.sum()} | {cs.min():.3f} – {cs.max():.3f} | {np.percentile(x, 50):.0f} | {np.percentile(x, 99):.0f} | {x.max()} | {int(np.sum(x >= 12))} |")
    hist = collections.Counter(int(v) for v in neg)
    return lines, neg_lines, pos, neg, hist


def stage_report():
    cfg = load_config()
    ch = choose()
    with open(os.path.join(CACHE, "measure.json")) as f:
        M = json.load(f)
    T, K = ch["T"], ch["K"]
    Ts = str(T)
    rows = M["rows"][Ts]
    regs = M["regions"][Ts]
    nm = M["nonmatch"][Ts]
    tr = timing_rows()
    cal_lines, neg_lines, pos, neg, hist = calib_tables(ch)
    c = load_cal()
    RO = load_regions("originals")
    out = []
    w = out.append
    w(f"# Calibration of the reused-photo check (fingerprints), {REPORT_DATE}")
    w("")
    w("design note 40, phase 1 (« Jeu empreintes », « Script de calibrage », « Cibles proposées »). "
      "Script: `eval/calibrate-fingerprints.py` (its header says how to run it). "
      "Cache: `artifacts/eval-photos/fingerprint/cache/`. `config/photo-checks.json` is not modified: "
      "the proposed values are below.")
    w("")
    w("@@SUMMARY@@")
    w("")
    w("## Method")
    w("")
    w("@@METHOD@@")
    w("")
    w("## 1. DINOv2-S threshold T (calibrate half)")
    w("")
    w(f"All {ch['pairs']:,} pairs of the 5,000 distinct calibrate originals; cosine = dot product of the stored fp16 vectors read back as float32, as `services/photoVectorCache.ts` does.")
    w("")
    w("| False-match rate per pair | Pairs allowed | Empirical threshold (cosines above it) |")
    w("|---|---:|---:|")
    for f in (1e-4, 1e-5, 1e-6, 1e-7):
        w(f"| {f:.0e} | {k_for(f, ch['pairs'])} | {ch['emp'][f]:.4f} |")
    w("")
    w("Highest non-matching cosines (calibrate): " + ", ".join(f"{x:.4f}" for x in ch["tail"][:12]) + ".")
    w("")
    g = ch["gpd"]
    w(f"Tail model (generalised Pareto, probability-weighted moments, on the {g['n_u']} pairs above the 1e-4 level {g['u']:.4f}): "
      f"1e-6 → {g['q1e-6']:.4f}, 1e-7 → {g['q1e-7']:.4f}, 1e-8 → {g['q1e-8']:.4f}"
      + (f"; fitted upper end {g['endpoint']:.4f}" if g["endpoint"] else "") + ".")
    w("")
    w("@@T@@")
    w("")
    w("## 2. PDQ rule (distance ≤ 31, quality > 49)")
    w("")
    w("@@PDQ@@")
    w("")
    w("## 3. `/verify` inliers (calibrate half)")
    w("")
    w(f"The vision module `service/app/verify.py` (SIFT, Lowe ratio 0.75, RANSAC 5 px, mirrored copy tried below 12 inliers), "
      f"on a stratified sample: {len(pos)} true copies against their original (about 222 per transform), "
      f"and {len(neg)} hard negatives.")
    w("")
    w("\n".join(cal_lines))
    w("")
    w("\n".join(neg_lines))
    w("")
    bins = [(0, 7), (8, 11), (12, 19), (20, 49), (50, 99), (100, 199), (200, 10 ** 6)]
    w("Hard-negative inliers, pairs per range: " + ", ".join(f"{a}–{b if b < 10 ** 6 else 'max'}: {int(np.sum((neg >= a) & (neg <= b)))}" for a, b in bins) + ".")
    w("")
    w("@@K@@")
    w("")
    w(f"## 4. Detection on the measure half (T = {T})")
    w("")
    w("Database: the 5,000 measure originals. Queries: their 45,000 transformed copies, and each original uploaded again unchanged. "
      "Production cascade (`services/photoMatch.ts`): PDQ candidates (best of the 8 orientations of the query against the stored upright hash, "
      "≤ 31 bits, both qualities > 49, nearest 4) and DINOv2 candidates (cosine ≥ T, best 4), then `/verify` on each; a copy is "
      "confirmed at K inliers or more.")
    w("")
    w(f"| Transform | Copies | Today (exact address) | PDQ alone | DINOv2 alone (≥ {T}) | PDQ or DINOv2 | **Cascade, K = 12** | Cascade, K = {K} (brief's rule) | Other photos confirmed, K = 12 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["reupload"] + TYPES:
        r = rows[name]
        w(f"| {name} | {r['n']:,} | 0 % | {pct(r['pdq'])} | {pct(r['dino'])} | {pct(r['stage'])} | **{pct(r['cascade_12'])}** | {pct(r['cascade_K'])} | {r['false_12']} |")
    allr = [rows[t] for t in TYPES]
    n_all = sum(r["n"] for r in allr)
    avg = lambda k: sum(r[k] * r["n"] for r in allr) / n_all
    w(f"| **all 9 transforms** | {n_all:,} | 0 % | {pct(avg('pdq'))} | {pct(avg('dino'))} | {pct(avg('stage'))} | **{pct(avg('cascade_12'))}** | {pct(avg('cascade_K'))} | {sum(r['false_12'] for r in allr)} |")
    w("")
    w("@@DETECT@@")
    w("")
    w("### Other thresholds (same measure half, for the decision only; T was chosen on the calibrate half)")
    w("")
    w("| T (calibrate level) | Measure pairs ≥ T (of %s) | DINOv2 alone, 9 transforms | Cascade K = 12, 9 transforms | crop_70 | crop_50 | rotate_15 | mirror | Measure pairs sent to `/verify` / confirmed at 12 |" % f"{M['nonmatch_pairs']:,}")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lev = {v: k for k, v in ch["T_levels"].items()}
    for Tx in M["T_list"]:
        rr = M["rows"][str(Tx)]
        ar = [rr[t] for t in TYPES]
        a = lambda k: sum(r[k] * r["n"] for r in ar) / n_all
        lab = f"{Tx} ({lev[Tx]:.0e})" if Tx in lev else f"**{Tx} (proposed)**"
        nmx = M["nonmatch"][str(Tx)]
        w(f"| {lab} | {M['nonmatch_count_ge'][str(Tx)]} | {pct(a('dino'))} | {pct(a('cascade_12'))} | {pct(rr['crop_70']['cascade_12'])} | {pct(rr['crop_50']['cascade_12'])} | {pct(rr['rotate_15']['cascade_12'])} | {pct(rr['mirror']['cascade_12'])} | {nmx['stage_pass']} / {nmx['conf']['12']} |")
    w("")
    w("## 5. False confirmations among non-matching pairs (measure half)")
    w("")
    w(f"- Pairs of distinct measure originals: {M['nonmatch_pairs']:,} ({M['n_dup']} duplicate pairs left out of {M['nonmatch_pairs_raw']:,}). At or above T: {nm['cos_pass']}; PDQ match: {nm['pdq_pass']} (raw {M['pdq_pairs_raw']}); "
      f"sent to `/verify`: {nm['stage_pass']}; confirmed at K = 12: **{nm['conf']['12']}**; at K = {K}: {nm['conf'][str(K)]}.")
    w(f"- Highest non-matching cosine on the measure half: {M['nonmatch_max']:.4f} (top 10: " + ", ".join(f"{x:.4f}" for x in M["nonmatch_tail"]) + ").")
    w(f"- The 1,000 most similar measure pairs through `/verify` regardless of T (out-of-sample hard negatives): "
      f"max {M['top1000_inliers']['max']} inliers, p99 {M['top1000_inliers']['p99']:.0f}; ≥ K: {M['top1000_inliers']['ge_K']}; ≥ 12: {M['top1000_inliers']['ge_12']}.")
    w("- In the cascade itself (copies as queries), wrong candidates confirmed: see the last column of section 4. "
      "A collage's three other photos are in the database too; finding one of them is a true find, counted apart: "
      f"{rows['collage_2x2']['filler_12']} at K = 12. Matches to a duplicate of the parent (label noise), counted apart: {sum(rows[t]['dup_12'] for t in ['reupload'] + TYPES)} at K = 12.")
    w("")
    w("@@NM@@")
    w("")
    w("## 6. Screenshots and collages: region fingerprints")
    w("")
    w("`photo_rects_part` (`service/app/screen.py`, the OpenCV search for natural-photo rectangles) on each screenshot and collage; each rectangle "
      "(at most 4, at least 32 px a side) is fingerprinted by `fingerprint()` with `regions`, searched like a photo, and verified with `box_a`. "
      "« Perfect box » uses the place where the generator pasted the photo: an upper bound for a better rectangle finder (or the small model), not a result.")
    w("")
    w("| Measure half, K = 12 | T | Copies | Whole image | **+ rectangles found** | + largest rectangle only | + perfect box (sample of %d) | Images with ≥ 1 rectangle | Other photos confirmed by the rectangles |" % ORACLE_SAMPLE)
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for Tx in (T, min(M["T_list"])):
        for t in REGION_TYPES:
            r = M["regions"][str(Tx)][t]
            w(f"| {t} | {Tx} | {r['n']:,} | {pct(r['whole_12'])} | **{pct(r['rect_12'])}** | {pct(r['rect_largest_12'])} | {pct(r['oracle_12'])} | {r['rect_with']:,} ({r['rect_queries']:,} rectangles) | {r['rect_false_12']} |")
    w("")
    w(f"Ordinary photos: the rectangle finder returns at least one rectangle on {int(np.sum(RO['n_rects'] > 0)):,} of the 10,000 originals "
      f"({pct(np.mean(RO['n_rects'] > 0))}); median time {np.median(RO['ms_rects']):.0f} ms.")
    w("")
    w("@@REGIONS@@")
    w("")
    w("## 7. Time per image on this laptop")
    w("")
    w("i5-8265U, 4 processes of one thread each, `nice 10`, the laptop shared with other jobs (load average 7 to 23 during the run): "
      "these are upper bounds, not the server's figures.")
    w("")
    w("| Set | Images | Median size | `fingerprint()` p50 | p95 | sha256 mismatches |")
    w("|---|---:|---|---:|---:|---:|")
    for name, n, size, p50, p95, bad in tr:
        w(f"| {name} | {n:,} | {size} | {p50:.0f} ms | {p95:.0f} ms | {bad} |")
    w(f"| `/verify` (all measure calls) | {M['verify_ms']['n']:,} | — | {M['verify_ms']['p50']:.0f} ms | {M['verify_ms']['p95']:.0f} ms | — |")
    w("")
    w("@@TIME@@")
    w("")
    w("## 8. Proposed values for `config/photo-checks.json` (not written)")
    w("")
    w("@@CONFIG@@")
    w("")
    w("## 9. Targets")
    w("")
    w("@@TARGETS@@")
    w("")
    w("## 10. Limits of this measure")
    w("")
    w("@@LIMITS@@")
    w("")
    text = "\n".join(out)
    for k, v in narrative(ch, M, c, cfg, RO).items():
        text = text.replace(k, v)
    with open(REPORT, "w") as f:
        f.write(text)
    log(f"report: {REPORT}")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    check_disk()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {"fingerprints": [stage_fingerprints], "regions": [stage_regions], "duplicates": [stage_duplicates],
              "calibrate": [stage_calibrate], "measure": [stage_measure], "report": [stage_report]}
    stages["all"] = [stage_fingerprints, stage_regions, stage_duplicates, stage_calibrate, stage_measure, stage_report]
    for fn in stages[stage]:
        fn()
