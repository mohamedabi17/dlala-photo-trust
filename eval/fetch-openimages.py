#!/usr/bin/env python3
"""The Open Images set of the « empreintes » calibration (design note 40, phase 4b): 4,000 distinct
real photos, mostly objects and indoor scenes, from the Open Images V7 validation split.

Sources, checked on 25/11/2026 on the official pages:
- https://storage.googleapis.com/openimages/web/download_v7.html: the image list is
  validation-images-with-rotation.csv (ImageID, OriginalURL, OriginalLandingURL, License,
  AuthorProfileURL, Author, ..., Rotation); human-verified image labels oidv7-val-annotations-human-imagelabels.csv;
  class names oidv7-class-descriptions.csv.
- https://raw.githubusercontent.com/openimages/dataset/master/downloader.py: bucket
  'open-images-dataset', key f'{split}/{image_id}.jpg', unsigned (anonymous) requests. The same
  object over HTTPS: https://open-images-dataset.s3.amazonaws.com/validation/<id>.jpg. The bucket
  serves copies at most 1,024 px a side (not the Flickr original: OriginalMD5 differs).
- https://storage.googleapis.com/openimages/web/factsfigures_v7.html: « The images are listed as
  having a CC BY 2.0 license »; annotations CC BY 4.0 (Google LLC).

Selection (seeded, no person looks at a photo):
1. licence field exactly CC BY 2.0, Rotation 0.0 (upright as served);
2. tier 1: a human-verified positive label among OBJECT_LABELS and none among PEOPLE_LABELS;
   tier 2: no people label; tier 3: the rest (not reached at 4,000);
3. at most one photo per Flickr author (no bursts of one scene by one photographer);
4. inside a tier, the order of sha256(SEED + id);
5. downloaded in that order; a photo within PDQ <= 8 bits (best of 8 orientations) of a photo
   already kept is dropped (near-duplicate), until COUNT are kept;
6. stored as JPEG quality 85, resized to MAX_SIDE only when larger (never: the bucket's copies are at
   most 1,024 px, and nothing is enlarged); the bucket's files are about 290 KB, 130 KB at quality 85;
7. split by a seeded shuffle: the first half « calibrate », the second « measure ».

Run (from the repository root, the vision venv of calibrate-openimages.py):
  python -I eval/fetch-openimages.py --meta <folder with the 3 CSV files>
Writes eval/openimages-manifest.json and <artifacts>/eval-photos/openimages/<id>.jpg.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
# The vision service's own modules (service/app), imported directly.
sys.path.insert(0, os.path.join(REPO, "service"))
ART = os.environ.get("EVAL_PHOTOS_DIR", os.path.join(REPO, "artifacts", "eval-photos"))
DEST = os.path.join(ART, "openimages")
MANIFEST = os.path.join(HERE, "openimages-manifest.json")

LICENSE = "https://creativecommons.org/licenses/by/2.0/"
CVDF = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
META_URLS = {
    "images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "labels": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
}
SEED = "dlala-openimages-20261125"
# Some Flickr author names are email addresses: the manifest never keeps one.
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def scrub_email(author):
    return EMAIL.sub("(address removed)", author or "")
COUNT = 4000
NEAR_DUP_BITS = 8
MAX_SIDE = 1280
JPEG_QUALITY = 85
DOWNLOAD_THREADS = 8
MIN_FREE_GB = 4.0
BATCH = 200

# What a seller photographs: goods and the rooms they stand in.
OBJECT_LABELS = {
    "Antique", "Backpack", "Bag", "Bathroom", "Bedroom", "Bench", "Bicycle", "Bookcase", "Bracelet", "Cabinetry", "Candle",
    "Ceramic", "Clock", "Coffee table", "Computer monitor", "Cosmetics", "Desk", "Doll", "Drawer", "Figurine", "Footwear",
    "Furniture", "Glasses", "Hammer", "Handbag", "Headphones", "Home appliance", "Interior design", "Jewellery", "Kettle",
    "Kitchen", "Kitchen & dining room table", "Kitchen appliance", "Kitchenware", "Lamp", "Lego", "Living room",
    "Luggage and bags", "Mirror", "Model car", "Mug", "Musical instrument", "Necklace", "Nightstand", "Office supplies",
    "Painting", "Perfume", "Picture frame", "Porcelain", "Pottery", "Ring", "Room", "Sandal", "Scooter", "Sewing machine",
    "Shelf", "Shoe", "Skateboard", "Sofa bed", "Sports equipment", "Stool", "Stuffed toy", "Suitcase", "Sunglasses",
    "Tablet computer", "Tableware", "Teapot", "Teddy bear", "Telephone", "Television", "Tool", "Toy", "Toy block",
    "Typewriter", "Umbrella", "Vase", "Wardrobe", "Watch", "Table", "Chair", "Couch", "Bed", "Chest of drawers",
    "Houseplant", "Flowerpot", "Book", "Bottle", "Glass", "Bowl", "Plate", "Coffee cup", "Wine glass", "Saucer", "Spoon",
    "Knife", "Camera", "Mobile phone", "Laptop", "Computer keyboard", "Guitar", "Piano", "Box", "Poster", "Sculpture",
    "Curtain", "Hat", "Boot", "High heels", "Dress", "Clothing", "Textile", "Motorcycle", "Ball (Object)", "Machine",
}
PEOPLE_LABELS = {
    "Man", "Woman", "Girl", "Boy", "Person", "Human", "Human face", "Human body", "Human head", "Human hand", "Human mouth",
    "Human ear", "Human eye", "Human nose", "Human hair", "Human leg", "Human arm", "Human foot", "Human beard", "Child",
    "Baby", "Toddler", "Bride", "Groom",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_gb():
    return shutil.disk_usage(ART).free / 1e9


def order_key(i):
    return hashlib.sha256((SEED + i).encode()).hexdigest()


def select(meta):
    names = {r[0]: r[1] for r in csv.reader(open(os.path.join(meta, "oidv7-class-descriptions.csv"), encoding="utf-8"))}
    pos = {}
    with open(os.path.join(meta, "oidv7-val-annotations-human-imagelabels.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Confidence"] == "1":
                pos.setdefault(r["ImageID"], set()).add(names.get(r["LabelName"], r["LabelName"]))
    rows = []
    with open(os.path.join(meta, "validation-images-with-rotation.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["License"] != LICENSE or r["Rotation"] != "0.0":
                continue
            labels = pos.get(r["ImageID"], set())
            tier = 1 if (labels & OBJECT_LABELS and not labels & PEOPLE_LABELS) else 2 if not labels & PEOPLE_LABELS else 3
            rows.append((tier, order_key(r["ImageID"]), r, sorted(labels & OBJECT_LABELS)))
    rows.sort(key=lambda x: (x[0], x[1]))
    seen, out = set(), []
    for tier, _, r, objs in rows:
        if r["AuthorProfileURL"] in seen:
            continue
        seen.add(r["AuthorProfileURL"])
        out.append((tier, r, objs))
    log(f"candidates: {len(out)} (one per author); tier 1: {sum(t == 1 for t, _, _ in out)}, tier 2: {sum(t == 2 for t, _, _ in out)}")
    return out


def fetch(image_id):
    req = urllib.request.Request(CVDF.format(image_id), headers={"User-Agent": "eval-photos-fetch/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network errors: retried, then the photo is skipped
            err = e
            time.sleep(1 + attempt)
    log(f"  {image_id}: download failed ({type(err).__name__})")
    return None


def prepare(data, stored):
    """(stored bytes, upright size, 8 PDQ hashes as (8, 4) uint64, quality) or None if not a usable image.
    stored: the bytes are this script's own file of an earlier run (kept as they are)."""
    import pdqhash
    from app import images
    try:
        im = images.decode(data)
    except images.BadInput:
        return None
    if not stored:
        im, _ = images.shrink(im, MAX_SIDE)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
    small, _ = images.shrink(im, 1024)
    vecs, q = pdqhash.compute_dihedral(np.asarray(small, np.uint8))
    h = np.stack([np.packbits(np.asarray(v, np.uint8)) for v in vecs]).view(">u8").astype(np.uint64)
    return data, list(im.size), h, int(q)


def min_dist(h8, kept_up):
    """Best distance over the 8 orientations of h8 against each kept upright hash."""
    if not len(kept_up):
        return np.array([], np.int32)
    best = None
    for o in range(8):
        d = sum(np.bitwise_count(h8[o, w] ^ kept_up[:, w]).astype(np.int32) for w in range(4))
        best = d if best is None else np.minimum(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="folder holding the three CSV files of META_URLS")
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    cands = select(a.meta)
    kept, kept_up, dropped, failed = [], np.zeros((0, 4), np.uint64), [], 0
    i = 0
    with cf.ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
        while len(kept) < a.count and i < len(cands):
            if free_gb() < MIN_FREE_GB:
                sys.exit(f"stop: {free_gb():.2f} GB free, under {MIN_FREE_GB} GB")
            batch = cands[i:i + BATCH]
            i += len(batch)
            path_of = lambda r: os.path.join(DEST, r["ImageID"] + ".jpg")
            have = [os.path.exists(path_of(c[1])) for c in batch]
            datas = list(ex.map(lambda c: open(path_of(c[1]), "rb").read() if os.path.exists(path_of(c[1])) else fetch(c[1]["ImageID"]), batch))
            for (tier, r, objs), data, stored in zip(batch, datas, have):
                if len(kept) >= a.count:
                    break
                p = None if data is None else prepare(data, stored)
                if p is None:
                    failed += 1
                    continue
                body, size, h, q = p
                d = min_dist(h, kept_up)
                if len(d) and d.min() <= NEAR_DUP_BITS:
                    j = int(d.argmin())
                    dropped.append({"id": r["ImageID"], "near": kept[j]["id"], "bits": int(d[j])})
                    if os.path.exists(path_of(r)):
                        os.remove(path_of(r))
                    continue
                with open(path_of(r), "wb") as f:
                    f.write(body)
                kept_up = np.vstack([kept_up, h[0:1]])
                kept.append({"id": r["ImageID"], "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "size": size,
                             "pdqQuality": q, "tier": tier, "objectLabels": objs, "license": r["License"], "author": scrub_email(r["Author"]),
                             "authorProfileURL": r["AuthorProfileURL"], "originalURL": r["OriginalURL"],
                             "landingURL": r["OriginalLandingURL"], "url": CVDF.format(r["ImageID"])})
            log(f"  kept {len(kept)} / {a.count} (looked at {i}; near-duplicates {len(dropped)}; failed {failed}; free {free_gb():.2f} GB)")
    rng = np.random.default_rng(int(hashlib.sha256(SEED.encode()).hexdigest()[:8], 16))
    perm = rng.permutation(len(kept))
    for rank, k in enumerate(perm):
        kept[k]["half"] = "calibrate" if rank < len(kept) // 2 else "measure"
    keep_ids = {x["id"] + ".jpg" for x in kept}
    for fn in os.listdir(DEST):  # files of an earlier run that are no longer selected
        if fn.endswith(".jpg") and fn not in keep_ids:
            os.remove(os.path.join(DEST, fn))
    out = {
        "about": "Open Images V7 validation photos for the reused-photo calibration (fetch-openimages.py). Files: <artifacts>/eval-photos/openimages/<id>.jpg; sha256 of the stored bytes.",
        "sources": META_URLS, "bucket": "open-images-dataset (CVDF), key validation/<id>.jpg, " + CVDF.format("<id>"),
        "licenseRule": "License field exactly " + LICENSE + "; annotations CC BY 4.0 Google LLC",
        "seed": SEED, "selection": "tier 1 objects/indoor without people, then tier 2 without people; one per author; seeded order; PDQ <= 8 near-duplicates dropped",
        "counts": {"kept": len(kept), "nearDuplicatesDropped": len(dropped), "failed": failed, "lookedAt": i,
                   "byTier": {str(t): sum(x["tier"] == t for x in kept) for t in (1, 2, 3)},
                   "byHalf": {h: sum(x["half"] == h for x in kept) for h in ("calibrate", "measure")}},
        "nearDuplicates": dropped, "images": kept,
    }
    with open(MANIFEST, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"manifest: {len(kept)} photos, {sum(x['bytes'] for x in kept) / 1e6:.0f} MB; {MANIFEST}")


if __name__ == "__main__":
    main()
