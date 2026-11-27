"""POST /verify: geometric proof that two photos are the same picture (plan phase 4).

SIFT is in OpenCV's main module since 4.4.0, its patent expired in March 2020, and the pinned
opencv-python-headless 4.14.0.94 is Apache-2.0 (https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html).
Matches use Lowe's ratio test, then cv2.findHomography(..., cv2.RANSAC); the inliers are the
proof. The benchmark of the plan (arXiv:2610.04151) counts 12 inliers as a match; the threshold
itself is the API's (config/photo-checks.json). A mirrored copy is tried too, since SIFT is not
mirror-invariant and PDQ's 8 orientations find mirrored copies.
"""
import base64

import cv2
import numpy as np

from .images import parse_box, shrink

# Both images are compared at most this large (speed; keypoints of a 4096 px original are mostly noise).
VERIFY_MAX_SIDE = 1024
MAX_FEATURES = 2000
# Lowe's ratio test (Lowe 2004, section 7.1: 0.8; 0.75 is the common stricter choice).
LOWE_RATIO = 0.75
# RANSAC reprojection error, in pixels of the compared copies.
RANSAC_REPROJ_PX = 5.0
# findHomography needs at least 4 point pairs.
MIN_MATCHES_FOR_HOMOGRAPHY = 4
# Below this many inliers the mirrored copy is also tried. It must stay above the API's verify.minInliers:
# a mirrored copy matched unmirrored keeps a few dozen chance inliers (phase 4b: median 45 on real photos),
# so a lower value would stop the mirrored try short of the API's threshold.
TRY_MIRROR_BELOW = 60
# A homography that crushes or folds the picture proves nothing: RANSAC can fit dozens of matches that all
# land on one spot of the other photo (measured in phase 4b on unrelated photos: 79 "inliers" at cosine
# 0.19). Kept only when the first picture's corners map to a convex quadrilateral and both its linear part
# and that quadrilateral's true area (which differs under strong perspective) scale areas by
# 1/MAX_SCALE**2 to MAX_SCALE**2; inliers count once per distinct point on each side. A rejected
# homography is not returned (no matrix, no lines on the evidence picture).
MAX_SCALE = 20.0
DISTINCT_PX = 2.0
# Evidence picture: both photos side by side at this height, inlier lines drawn, small JPEG.
EVIDENCE_HEIGHT = 320
EVIDENCE_JPEG_QUALITY = 70
MAX_EVIDENCE_LINES = 60


def _prep(im, box):
    if box is not None:
        im = im.crop(tuple(box))
    small, scale = shrink(im, VERIFY_MAX_SIDE)
    return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)


def _features(sift, g):
    kp, des = sift.detectAndCompute(g, None)
    return {"kp": kp, "des": des, "shape": g.shape[:2]}


def _sane(H, shape_a):
    if H is None or abs(H[2, 2]) < 1e-12:
        return False
    H = H / H[2, 2]
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if not (1.0 / MAX_SCALE ** 2 <= abs(det) <= MAX_SCALE ** 2):
        return False
    h, w = shape_a
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not (np.all(np.isfinite(quad)) and cv2.isContourConvex(quad.astype(np.float32))):
        return False
    area = cv2.contourArea(quad.astype(np.float32)) / float(w * h)
    return 1.0 / MAX_SCALE ** 2 <= area <= MAX_SCALE ** 2


def _reflects(H):
    """The homography mirrors the picture (its linear part turns it over)."""
    if H is None or abs(H[2, 2]) < 1e-12:
        return False
    H = H / H[2, 2]
    return bool(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0] < 0)


def _distinct(points):
    return len({(int(x // DISTINCT_PX), int(y // DISTINCT_PX)) for x, y in points})


def _match_features(fa, fb):
    ka, da, kb, db = fa["kp"], fa["des"], fb["kp"], fb["des"]
    res = {"ka": ka, "kb": kb, "good": [], "mask": None, "H": None, "inliers": 0}
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return res
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]
    res["good"] = good
    if len(good) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return res
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX)
    if H is not None and mask is not None and _sane(H, fa["shape"]):
        res["H"], res["mask"] = H, mask.ravel().astype(bool)
        ok = res["mask"]
        res["inliers"] = min(_distinct(src.reshape(-1, 2)[ok]), _distinct(dst.reshape(-1, 2)[ok]))
    return res


def _inliers(res):
    return res["inliers"]


def _evidence(ga, gb, res):
    def fit(g):
        s = EVIDENCE_HEIGHT / g.shape[0]
        return cv2.resize(g, (max(1, round(g.shape[1] * s)), EVIDENCE_HEIGHT), interpolation=cv2.INTER_AREA), s

    a, sa = fit(ga)
    b, sb = fit(gb)
    canvas = cv2.cvtColor(np.hstack([a, b]), cv2.COLOR_GRAY2BGR)
    if res["mask"] is not None:
        inl = [m for m, ok in zip(res["good"], res["mask"]) if ok][:MAX_EVIDENCE_LINES]
        for m in inl:
            pa = res["ka"][m.queryIdx].pt
            pb = res["kb"][m.trainIdx].pt
            p1 = (int(pa[0] * sa), int(pa[1] * sa))
            p2 = (int(pb[0] * sb) + a.shape[1], int(pb[1] * sb))
            cv2.line(canvas, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, (0, 0, 255), -1)
            cv2.circle(canvas, p2, 2, (0, 0, 255), -1)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def verify(im_a, im_b, meta):
    box_a = meta.get("box_a")
    box_b = meta.get("box_b")
    box_a = None if box_a is None else parse_box(box_a, *im_a.size, name="box_a")
    box_b = None if box_b is None else parse_box(box_b, *im_b.size, name="box_b")
    ga, gb = _prep(im_a, box_a), _prep(im_b, box_b)
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    fa = _features(sift, ga)
    res, flipped = _match_features(fa, _features(sift, gb)), False
    # The mirrored second image is tried when few inliers were found, or when the kept homography is itself
    # a reflection (symmetric content matched turned over: the mirrored try usually finds many more).
    if _inliers(res) < TRY_MIRROR_BELOW or _reflects(res["H"]):
        gb_m = np.ascontiguousarray(gb[:, ::-1])
        res_m = _match_features(fa, _features(sift, gb_m))
        if _inliers(res_m) > _inliers(res):
            res, gb, flipped = res_m, gb_m, True
    return {
        "inliers": _inliers(res),
        "matches": len(res["good"]),
        "keypoints": [len(res["ka"] or []), len(res["kb"] or [])],
        # mirrored: the copy is a mirror image (the mirrored try won, or the kept homography turns it over)
        "mirrored": bool(flipped != _reflects(res["H"])),
        "homography": None if res["H"] is None else [[round(float(v), 6) for v in row] for row in res["H"]],
        "method": "sift+ransac",
        "evidence_jpeg": _evidence(ga, gb, res),
    }
