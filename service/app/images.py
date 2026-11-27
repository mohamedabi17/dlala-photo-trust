"""Decoding of untrusted image bytes, size limits, boxes and the shared preprocessing."""
import io
import math
import numbers

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Phase 2 keeps originals at most 4096 px a side, so 4096 x 4096 is the largest real input.
MAX_IMAGE_PIXELS = 4096 * 4096
# A very long thin image is refused even under the pixel limit.
MAX_SIDE = 8192
# Below this the models have nothing to look at.
MIN_SIDE = 16
# The only formats Pillow may parse here: phase 2's originals are JPEG, and phones send JPEG,
# PNG or WebP. TIFF, BMP, GIF and the rest are never parsed. MPO (a JPEG with several frames) has
# no opener of its own: Pillow's JPEG opener returns it, so it is allowed after opening only.
OPEN_FORMATS = ["JPEG", "PNG", "WEBP"]
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "MPO")
# Pillow refuses to decode above this (decompression bombs); our own check comes first.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS * 2


class BadInput(Exception):
    """A request the service refuses: (HTTP status, machine code, human message)."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def decode(data):
    """Bytes to an RGB Pillow image, upright (EXIF orientation applied). Raises BadInput."""
    if not data:
        raise BadInput(400, "empty_image", "the image part is empty")
    try:
        im = Image.open(io.BytesIO(data), formats=OPEN_FORMATS)
    except Image.DecompressionBombError:  # Pillow checks the header's size already when opening
        raise BadInput(413, "image_too_large", f"the image is above the limit of {MAX_IMAGE_PIXELS} pixels")
    except (UnidentifiedImageError, OSError, ValueError):
        raise BadInput(415, "not_an_image", "the bytes are not an image this service reads")
    if im.format not in ALLOWED_FORMATS:
        raise BadInput(415, "not_an_image", f"unsupported image format {im.format}")
    w, h = im.size
    if w * h > MAX_IMAGE_PIXELS or max(w, h) > MAX_SIDE:
        raise BadInput(413, "image_too_large", f"{w}x{h} is above the limit of {MAX_IMAGE_PIXELS} pixels or {MAX_SIDE} px a side")
    if min(w, h) < MIN_SIDE:
        raise BadInput(400, "image_too_small", f"{w}x{h} is below {MIN_SIDE} px a side")
    try:
        im.load()
        # In place and convert only when needed: each copy of a 4096 px original is 50 MB.
        ImageOps.exif_transpose(im, in_place=True)
        return im if im.mode == "RGB" else im.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise BadInput(415, "not_an_image", "the image is damaged or truncated")


def to_rgb_array(im):
    return np.asarray(im, dtype=np.uint8)


def to_bgr_array(im):
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8)[:, :, ::-1])


def shrink(im, max_side):
    """A copy whose longest side is at most max_side (area filter), or the image itself."""
    w, h = im.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return im, 1.0
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BOX), scale


def parse_box(box, w, h, name="box"):
    """[x0, y0, x1, y1] in pixels of the upright image (x1, y1 exclusive), clamped to it. Raises BadInput."""
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        raise BadInput(400, "bad_meta", f"{name} must be [x0, y0, x1, y1]: four finite numbers, in pixels")
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise BadInput(400, "bad_meta", f"{name} {list(box)} is empty inside a {w}x{h} image")
    return [x0, y0, x1, y1]


def to_tensor(im, size, resample, mean, std, box=None):
    """The image (or its box) resized to size x size (no crop), scaled to [0, 1], normalised, CHW float32."""
    x = np.asarray(im.resize((size, size), resample, box=None if box is None else tuple(box)), dtype=np.float32) * np.float32(1 / 255)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))
