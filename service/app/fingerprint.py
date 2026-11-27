"""POST /fingerprint: PDQ (8 orientations and quality) and a DINOv2 vector, for the image and its regions."""
import base64

import numpy as np
import pdqhash

from .images import BadInput, parse_box, shrink, to_rgb_array

# PDQ reduces every image to 64 x 64 luma anyway; hashing a copy of at most this size keeps the
# time flat on 4096 px originals. The test suite measures the distance this costs (tests/test_pdq.py).
PDQ_MAX_SIDE = 1024
# Regions (photo rectangles found in a screenshot, phase 4): at most this many per request.
MAX_REGIONS = 8
# A region smaller than this on a side carries too little to fingerprint.
MIN_REGION_SIDE = 32
# Query tiles (phase 4b, calibration-fingerprints-2026-11-25.md): with "tiles": true, the 2 x 2 quadrants
# and the centre half are fingerprinted like regions (a photo pasted in a collage or a page is often one of
# them), after the asked regions, marked "tile": true.
def tile_boxes(w, h):
    hw, hh = w // 2, h // 2
    return [[0, 0, hw, hh], [hw, 0, w, hh], [0, hh, hw, h], [hw, hh, w, h], [w // 4, h // 4, w - w // 4, h - h // 4]]


# compute_dihedral order (pdqhash bindings.pyx; same as ThreatExchange's pdq-photo-hasher --pdqdih).
ORIENTATIONS = ["original", "rotate90", "rotate180", "rotate270", "flipx", "flipy", "flipplus1", "flipminus1"]


def bits_to_hex(vector):
    """pdqhash's 256 bits (most significant first) as ThreatExchange's 64-character hex."""
    return np.packbits(np.asarray(vector, dtype=np.uint8)).tobytes().hex()


def pdq(im):
    small, _ = shrink(im, PDQ_MAX_SIDE)
    vectors, quality = pdqhash.compute_dihedral(to_rgb_array(small))
    return [bits_to_hex(v) for v in vectors], int(quality)


def fp16_b64(vec):
    return base64.b64encode(np.asarray(vec, dtype="<f2").tobytes()).decode("ascii")


def fingerprint(reg, im, meta):
    regions = meta.get("regions") or []
    if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
        raise BadInput(400, "bad_meta", f"regions must be a list of at most {MAX_REGIONS} boxes")
    tiles = meta.get("tiles", False)
    if not isinstance(tiles, bool):
        raise BadInput(400, "bad_meta", "tiles must be true or false")
    w, h = im.size
    boxes = [parse_box(b, w, h, f"regions[{i}]") for i, b in enumerate(regions)]
    for i, b in enumerate(boxes):
        if min(b[2] - b[0], b[3] - b[1]) < MIN_REGION_SIDE:
            raise BadInput(400, "bad_meta", f"regions[{i}] is smaller than {MIN_REGION_SIDE} px a side")
    asked = len(boxes)
    if tiles and min(w, h) >= 2 * MIN_REGION_SIDE:
        boxes += tile_boxes(w, h)
    vecs = reg.dino_embed([im], boxes)
    hashes, quality = pdq(im)
    out = {
        "pdq": hashes, "quality": quality, "orientations": ORIENTATIONS,
        "dino": fp16_b64(vecs[0]), "dino_dim": int(vecs.shape[1]), "dino_variant": reg.dino_variant,
        "size": [w, h], "regions": [],
    }
    for k, (b, v) in enumerate(zip(boxes, vecs[1:])):
        rh, rq = pdq(im.crop(tuple(b)))
        out["regions"].append({"box": b, "pdq": rh, "quality": rq, "dino": fp16_b64(v), "tile": k >= asked})
    return out
