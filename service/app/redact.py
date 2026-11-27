"""POST /redact: a strong Gaussian blur on the boxes phase 5 kept (faces of real people, ID text,
plates), each box enlarged by 15 % (plan phase 3). Returns the whole photo as a JPEG."""
import io

import cv2
import numpy as np
from PIL import Image

from .images import BadInput, parse_box

# "cadre agrandi de 15 %": width and height grow by 15 %, around the same centre.
EXPAND = 0.15
MAX_BOXES = 32
# The blur's sigma as a fraction of the box's longer side: a face becomes a smooth blob.
SIGMA_FRACTION = 0.25
# The blur is computed on a copy of the box shrunk to this many pixels on its longer side, then
# enlarged back: the same strong Gaussian, in constant time whatever the box size.
WORK_SIDE = 64
JPEG_QUALITY = 90


def expand(box, w, h):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * EXPAND / 2, (y1 - y0) * EXPAND / 2
    return [max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(np.ceil(x1 + dx))), min(h, int(np.ceil(y1 + dy)))]


def blur_region(region):
    rh, rw = region.shape[:2]
    s = WORK_SIDE / max(rw, rh)
    small = cv2.resize(region, (max(1, round(rw * s)), max(1, round(rh * s))), interpolation=cv2.INTER_AREA)
    sigma = max(small.shape[:2]) * SIGMA_FRACTION
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    big = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    # A second pass at full size removes the faint grid of the enlargement.
    return cv2.GaussianBlur(big, (0, 0), sigmaX=max(1.0, max(rw, rh) / WORK_SIDE))


def redact(im, meta):
    boxes = meta.get("boxes")
    if not isinstance(boxes, list) or not boxes or len(boxes) > MAX_BOXES:
        raise BadInput(400, "bad_meta", f"boxes must be a list of 1 to {MAX_BOXES} boxes")
    w, h = im.size
    arr = np.array(im)  # RGB, writable copy
    done = []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = expand(parse_box(b, w, h, f"boxes[{i}]"), w, h)
        arr[y0:y1, x0:x1] = blur_region(arr[y0:y1, x0:x1])
        done.append([x0, y0, x1, y1])
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), done
