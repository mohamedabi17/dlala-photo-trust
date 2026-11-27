"""The copies a fraudster makes of someone else's photo (note 40, phase 4b), made in memory with Pillow.

Same 9 types as lib/transforms.mjs (phase 1), with what that set lacked: crops at a random place
(not only centred), screenshots of varied synthetic classifieds pages (page and background colour, white
included; photo size and place; letterbox; thumbnails of other photos; phone and desktop layouts), and
2 x 2 collages with or without gutters, the photo in a random cell. Nothing is written to disk: the
calibration regenerates a copy whenever it needs its pixels.

Deterministic: every random choice comes from a generator seeded by sha256(SEED | parent id | type);
same Pillow, libjpeg and DejaVu fonts give the same bytes. Site names use the reserved .example domain.
"""
import hashlib
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SEED = "dlala-transforms-20261125"
TYPES = ["jpeg_q50", "downscale_50", "crop_70", "crop_50", "rotate_15", "mirror", "text_overlay", "classifieds_page", "collage_2x2"]
REGION_TYPES = ["classifieds_page", "collage_2x2"]
OUT_QUALITY = 90
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
OVERLAY_PHRASES = ["À vendre", "Très bon état", "Prix : 45 €", "Envoi possible", "Photo non contractuelle", "Neuf, jamais servi",
                   "Contact en message privé", "Remise en main propre"]
OVERLAY_TEXT_SHARE = 0.09
WORDS = ("ancien vintage très bon état lot complet authentique rare signé collection objet décoration maison bois laiton "
         "porcelaine cuir original livraison possible remise en main propre prix ferme négociable dimensions hauteur "
         "largeur poids marque modèle année fonctionne parfaitement quelques traces usage voir photos").split()
TITLES = ["Lampe ancienne en laiton", "Vase en porcelaine", "Fauteuil vintage", "Appareil photo argentique", "Lot de vaisselle",
          "Sac en cuir", "Montre de collection", "Guitare acoustique", "Table basse en bois", "Jouet ancien", "Vélo de ville",
          "Miroir doré", "Service à thé", "Commode années 60"]
BUTTONS = ["Contacter", "Acheter", "Faire une offre", "Message", "Réserver"]
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    return _fonts[key]


def rng_for(parent, kind):
    return np.random.default_rng(int(hashlib.sha256(f"{SEED}|{parent}|{kind}".encode()).hexdigest()[:16], 16))


def jpeg(im, quality=OUT_QUALITY):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pick_fillers(rng, pool, parent, k):
    out = []
    while len(out) < k:
        f = pool[int(rng.integers(len(pool)))]
        if f != parent and f not in out:
            out.append(f)
    return out


def window(rng, w, h, side):
    cw, ch = round(w * side), round(h * side)
    x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return (x, y, x + cw, y + ch)


def _colour(rng, lo, hi):
    return tuple(int(v) for v in rng.integers(lo, hi + 1, 3))


def _page(rng, photo, fillers_im):
    """A synthetic classifieds page with the photo; returns (page, box of the photo's pixels, thumbnail boxes)."""
    phone = rng.random() < 0.65
    if phone:
        W, H = 720, int(rng.integers(1280, 1601))
    else:
        W, H = [(1280, 800), (1366, 768), (1440, 900)][int(rng.integers(3))]
    r = rng.random()
    if r < 0.40:
        bg = (255, 255, 255)
    elif r < 0.65:
        g = int(rng.integers(236, 249)); bg = (g, g, min(255, g + 2))
    elif r < 0.80:
        g = int(rng.integers(14, 40)); bg = (g, g, g + 2)
    else:
        bg = _colour(rng, 215, 250)
    dark = sum(bg) < 300
    ink = (229, 231, 235) if dark else _colour(rng, 15, 60)
    soft = (90, 95, 105) if dark else (200, 204, 210)
    page = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(page)
    y = 0
    if phone:  # status bar
        d.text((24, 10), f"{int(rng.integers(7, 23))}:{int(rng.integers(0, 60)):02d}", font=font(22, True), fill=ink)
        d.text((W - 110, 10), f"{int(rng.integers(10, 100))} %", font=font(22, True), fill=ink)
        y = 44
    hh = int(rng.integers(56, 97))
    head = bg if rng.random() < 0.3 else _colour(rng, 20, 200)
    d.rectangle([0, y, W, y + hh], fill=head)
    head_ink = (255, 255, 255) if sum(head) < 400 else (20, 20, 20)
    d.text((20, y + hh // 2 - 14), "annonces.example", font=font(26, True), fill=head_ink)
    if not phone:
        d.rounded_rectangle([W - 420, y + 14, W - 40, y + hh - 14], radius=14, fill=(255, 255, 255), outline=soft)
        d.text((W - 400, y + hh // 2 - 11), "Rechercher", font=font(20), fill=(120, 120, 120))
    if head == bg:
        d.line([0, y + hh, W, y + hh], fill=soft, width=2)
    y += hh + int(rng.integers(0, 40))
    pw, ph = photo.size
    ar = pw / ph
    if phone:
        bw = W if rng.random() < 0.5 else W - 2 * int(rng.integers(12, 33))
        x0 = (W - bw) // 2
        max_h = int(H * 0.6)
    else:
        bw = int(W * rng.uniform(0.42, 0.62))
        x0 = int(rng.integers(24, 81))
        max_h = H - y - 30
    letter = rng.random() < 0.4
    if letter:
        box_ar = [4 / 3, 1.0, 16 / 9][int(rng.integers(3))]
        bh = int(bw / box_ar)
    else:
        bh = int(bw / ar)
    if bh > max_h:
        bw, bh = int(bw * max_h / bh), max_h
    if letter:
        lb = [(0, 0, 0), (255, 255, 255), (238, 238, 238), bg][int(rng.integers(4))]
        d.rectangle([x0, y, x0 + bw - 1, y + bh - 1], fill=lb)
        scale = min(bw / pw, bh / ph)
        iw, ih = max(1, round(pw * scale)), max(1, round(ph * scale))
        ix, iy = x0 + (bw - iw) // 2, y + (bh - ih) // 2
    else:
        iw, ih, ix, iy = bw, bh, x0, y
    page.paste(photo.resize((iw, ih), Image.Resampling.LANCZOS), (ix, iy))
    photo_box = [ix, iy, ix + iw, iy + ih]
    # text: under the photo (phone) or in the right column (desktop)
    if phone:
        tx, ty, tw = 24, y + bh + int(rng.integers(14, 40)), W - 48
    else:
        tx, ty, tw = x0 + bw + int(rng.integers(30, 70)), y + 10, W - (x0 + bw) - 100
    title = TITLES[int(rng.integers(len(TITLES)))]
    d.text((tx, ty), title, font=font(int(rng.integers(26, 37)), True), fill=ink); ty += 50
    d.text((tx, ty), f"{int(rng.integers(5, 900))} €", font=font(int(rng.integers(30, 41)), True), fill=ink); ty += 56
    for _ in range(int(rng.integers(3, 8))):
        line = " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), int(rng.integers(4, 9))))
        d.text((tx, ty), line, font=font(int(rng.integers(18, 23))), fill=ink if rng.random() < 0.7 else (120, 120, 128)); ty += 30
    thumbs = []
    if fillers_im and rng.random() < 0.35:
        ty += 16
        n = len(fillers_im)
        ts = int((tw - (n - 1) * 12) / n) if phone else int(min(160, tw / n - 12))
        for k, f in enumerate(fillers_im):
            bx = tx + k * (ts + 12)
            if ty + ts < H - 10 and bx + ts < W:
                page.paste(ImageOps.fit(f, (ts, ts), Image.Resampling.LANCZOS), (bx, ty))
                thumbs.append([bx, ty, bx + ts, ty + ts])
        ty += ts + 16
    for k in range(int(rng.integers(1, 3))):
        bw2 = 200
        bx, by = tx + k * (bw2 + 20), min(H - 70, ty + 10)
        fill = _colour(rng, 20, 240)
        d.rounded_rectangle([bx, by, bx + bw2, by + 48], radius=10, fill=fill)
        d.text((bx + 20, by + 12), BUTTONS[int(rng.integers(len(BUTTONS)))], font=font(20, True),
               fill=(255, 255, 255) if sum(fill) < 400 else (20, 20, 20))
    if phone and rng.random() < 0.5:  # bottom navigation bar
        d.rectangle([0, H - 64, W, H], fill=bg if rng.random() < 0.5 else soft)
        for k in range(5):
            cx = int(W * (k + 0.5) / 5)
            d.ellipse([cx - 12, H - 44, cx + 12, H - 20], outline=ink, width=3)
    return page, photo_box, thumbs


def _collage(rng, photo, fillers_im):
    pw, ph = photo.size
    ar = pw / ph
    TW = int(rng.integers(1080, 1601))
    g = 0 if rng.random() < 0.5 else int(rng.integers(4, 21))
    gc = (255, 255, 255) if rng.random() < 0.5 else (0, 0, 0) if rng.random() < 0.5 else _colour(rng, 0, 255)
    cw = (TW - 3 * g) // 2
    ch = max(16, round(cw / ar))
    W, H = 2 * cw + 3 * g, 2 * ch + 3 * g
    out = Image.new("RGB", (W, H), gc)
    pos = int(rng.integers(4))
    cells = list(fillers_im)
    cells.insert(pos, None)
    box = None
    for k, f in enumerate(cells):
        x, y = g + (k % 2) * (cw + g), g + (k // 2) * (ch + g)
        if f is None:
            out.paste(photo.resize((cw, ch), Image.Resampling.LANCZOS), (x, y))
            box = [x, y, x + cw, y + ch]
        else:
            out.paste(ImageOps.fit(f, (cw, ch), Image.Resampling.LANCZOS), (x, y))
    return out, box, pos


def make(parent_id, kind, photo, pool, load):
    """JPEG bytes of the copy and what is known about it: box (where the photo is, screenshots and
    collages), window (crops), fillers (other photos shown: collage cells, page thumbnails).
    photo: the parent as an upright RGB PIL image; pool: ids of the same half; load(id): RGB PIL image."""
    rng = rng_for(parent_id, kind)
    w, h = photo.size
    info = {"type": kind, "parent": parent_id}
    if kind == "jpeg_q50":
        return jpeg(photo, 50), info
    if kind == "downscale_50":
        return jpeg(photo.resize((max(1, round(w / 2)), max(1, round(h / 2))), Image.Resampling.LANCZOS)), info
    if kind in ("crop_70", "crop_50"):
        b = window(rng, w, h, 0.7 if kind == "crop_70" else 0.5)
        info["window"] = list(b)
        return jpeg(photo.crop(b)), info
    if kind == "rotate_15":
        return jpeg(photo.rotate(-15, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))), info
    if kind == "mirror":
        return jpeg(photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)), info
    if kind == "text_overlay":
        im = photo.copy()
        d = ImageDraw.Draw(im)
        text = OVERLAY_PHRASES[int(rng.integers(len(OVERLAY_PHRASES)))]
        size = max(12, round(h * OVERLAY_TEXT_SHARE))
        f = font(size, True)
        tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2, h - size * 1.6), text, font=f, fill=(255, 255, 255), stroke_width=max(1, size // 12), stroke_fill=(0, 0, 0))
        return jpeg(im), info
    if kind == "classifieds_page":
        fill = pick_fillers(rng, pool, parent_id, int(rng.integers(3, 5)))
        page, box, thumbs = _page(rng, photo, [load(f) for f in fill])
        info.update(box=box, thumbs=thumbs, fillers=fill[:len(thumbs)], size=list(page.size))
        return jpeg(page), info
    if kind == "collage_2x2":
        fill = pick_fillers(rng, pool, parent_id, 3)
        out, box, pos = _collage(rng, photo, [load(f) for f in fill])
        info.update(box=box, position=pos, fillers=fill, size=list(out.size))
        return jpeg(out), info
    raise ValueError(kind)
