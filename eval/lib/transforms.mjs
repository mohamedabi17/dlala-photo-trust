// The copies a fraudster makes of someone else's photo (plan 40, phase 1, « jeu empreintes »), as
// `sharp` operations. Each one is deterministic: same input, same sharp/libvips and fonts, same
// bytes. Random choices (phrase, price, collage position and fillers) come from the photo's own
// seeded hash, never from Math.random.
import sharp from 'sharp';
import { unitRandoms } from './common.mjs';

/** Every output is re-encoded at this JPEG quality, so the only strong recompression is jpeg_q50's. */
const OUT_QUALITY = 90;
const WHITE = { r: 255, g: 255, b: 255, alpha: 1 };
const FONT = 'DejaVu Sans, Noto Sans, sans-serif';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const jpeg = (img, quality = OUT_QUALITY) => img.jpeg({ quality }).toBuffer();

/** What sellers write on their photos: a reused photo often carries one of these. */
const OVERLAY_PHRASES = ['À vendre', 'Très bon état', 'Prix : 45 €', 'Envoi possible', 'Photo non contractuelle', 'Neuf, jamais servi', 'Contact en message privé', 'Remise en main propre'];
/** Text height as a share of the photo's height: readable, like a phone's text tool. */
const OVERLAY_TEXT_SHARE = 0.09;

/**
 * Fake classifieds page: a screenshot-like canvas. Sizes are those of a phone screenshot scaled
 * to half (a 960 px wide screen shown at 480 px); the photo box is 4:3 like most listing pages.
 * The site name uses the reserved .example domain: it imitates no real site.
 */
const PAGE = { width: 480, height: 680, header: 52, margin: 24, photoW: 432, photoH: 324 };
/** Each cell of the 2×2 collage, in px: 256 px photos shrunk to 3/4, as a 4-photo post shows them. */
const COLLAGE_CELL = 192;

async function fitInto(buf, width, height) {
  return sharp(buf).resize(width, height, { fit: 'contain', background: WHITE }).toBuffer();
}

function overlaySvg(width, height, text) {
  const size = Math.max(12, Math.round(height * OVERLAY_TEXT_SHARE));
  const y = height - Math.round(size * 0.6);
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
<text x="${width / 2}" y="${y}" text-anchor="middle" font-family="${FONT}" font-weight="bold" font-size="${size}" fill="#ffffff" stroke="#000000" stroke-width="${Math.max(1, size / 12)}" paint-order="stroke">${esc(text)}</text></svg>`);
}

function pageSvg(price) {
  const P = PAGE;
  const top = P.header + P.margin + P.photoH + 20;
  const line = (y, w) => `<rect x="${P.margin}" y="${y}" width="${w}" height="12" rx="6" fill="#d5d8de"/>`;
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${P.width}" height="${P.height}">
<rect width="100%" height="100%" fill="#f3f4f6"/>
<rect width="100%" height="${P.header}" fill="#1f5f8b"/>
<text x="${P.margin}" y="34" font-family="${FONT}" font-weight="bold" font-size="20" fill="#ffffff">annonces.example</text>
<rect x="${P.width - 150}" y="12" width="126" height="28" rx="14" fill="#ffffff" opacity="0.9"/>
<text x="${P.width - 87}" y="31" text-anchor="middle" font-family="${FONT}" font-size="13" fill="#1f5f8b">Rechercher</text>
<rect x="${P.margin - 2}" y="${P.header + P.margin - 2}" width="${P.photoW + 4}" height="${P.photoH + 4}" fill="#ffffff" stroke="#d5d8de"/>
${line(top, 300)}${line(top + 22, 220)}
<text x="${P.margin}" y="${top + 66}" font-family="${FONT}" font-weight="bold" font-size="26" fill="#111827">${esc(price)}</text>
${line(top + 84, 380)}${line(top + 104, 340)}${line(top + 124, 260)}
<rect x="${P.margin}" y="${P.height - 70}" width="200" height="44" rx="8" fill="#ff6e14"/>
<text x="${P.margin + 100}" y="${P.height - 42}" text-anchor="middle" font-family="${FONT}" font-weight="bold" font-size="17" fill="#ffffff">Contacter</text>
<rect x="${P.width - P.margin - 200}" y="${P.height - 70}" width="200" height="44" rx="8" fill="#ffffff" stroke="#1f5f8b" stroke-width="2"/>
<text x="${P.width - P.margin - 100}" y="${P.height - 42}" text-anchor="middle" font-family="${FONT}" font-weight="bold" font-size="17" fill="#1f5f8b">Acheter</text>
</svg>`);
}

/**
 * The transforms, in the order of the brief. `apply(buf, ctx)` returns JPEG bytes; ctx = { id, fillers }
 * where fillers are three other photos' buffers from the same half (collage only).
 */
export const TRANSFORMS = [
  { type: 'jpeg_q50', params: { quality: 50 }, about: 'JPEG recompression at quality 50',
    apply: (buf) => jpeg(sharp(buf), 50) },
  { type: 'downscale_50', params: { scale: 0.5 }, about: 'downscaled to 50% of each side',
    apply: async (buf) => { const m = await sharp(buf).metadata(); return jpeg(sharp(buf).resize(Math.round(m.width * 0.5))); } },
  { type: 'crop_70', params: { side: 0.7 }, about: 'centre crop keeping 70% of the width and of the height',
    apply: (buf) => centreCrop(buf, 0.7) },
  { type: 'crop_50', params: { side: 0.5 }, about: 'centre crop keeping 50% of the width and of the height',
    apply: (buf) => centreCrop(buf, 0.5) },
  { type: 'rotate_15', params: { degrees: 15, background: 'white', canvas: 'expanded' }, about: 'rotated 15° clockwise on a white, enlarged canvas',
    apply: (buf) => jpeg(sharp(buf).rotate(15, { background: WHITE })) },
  { type: 'mirror', params: { axis: 'horizontal' }, about: 'mirrored left to right',
    apply: (buf) => jpeg(sharp(buf).flop()) },
  { type: 'text_overlay', params: { phrases: OVERLAY_PHRASES, textHeight: OVERLAY_TEXT_SHARE }, about: 'a seller-style caption written across the bottom',
    apply: async (buf, ctx) => {
      const m = await sharp(buf).metadata();
      const [u] = unitRandoms('overlay', ctx.id, 1);
      return jpeg(sharp(buf).composite([{ input: overlaySvg(m.width, m.height, OVERLAY_PHRASES[Math.floor(u * OVERLAY_PHRASES.length)]) }]));
    } },
  { type: 'classifieds_page', params: PAGE, about: 'pasted into a fake classifieds-site page (header, price, buttons), like a screenshot',
    apply: async (buf, ctx) => {
      const [u] = unitRandoms('price', ctx.id, 1);
      const price = `${10 + Math.floor(u * 490)} €`; // a plausible price between 10 and 499 €
      const photo = await fitInto(buf, PAGE.photoW, PAGE.photoH);
      return jpeg(sharp(pageSvg(price)).composite([{ input: photo, left: PAGE.margin, top: PAGE.header + PAGE.margin }]));
    } },
  { type: 'collage_2x2', params: { cell: COLLAGE_CELL, fillersFrom: 'same half' }, about: 'one cell of a 2×2 collage; the three other cells are other photos of the same half',
    apply: async (buf, ctx) => {
      const cells = [...ctx.fillers];
      cells.splice(ctx.position, 0, buf);
      const tiles = await Promise.all(cells.map((b) => fitInto(b, COLLAGE_CELL, COLLAGE_CELL)));
      return jpeg(sharp({ create: { width: 2 * COLLAGE_CELL, height: 2 * COLLAGE_CELL, channels: 3, background: WHITE } })
        .composite(tiles.map((input, i) => ({ input, left: (i % 2) * COLLAGE_CELL, top: Math.floor(i / 2) * COLLAGE_CELL }))));
    } },
];

async function centreCrop(buf, side) {
  const m = await sharp(buf).metadata();
  const width = Math.round(m.width * side);
  const height = Math.round(m.height * side);
  return jpeg(sharp(buf).extract({ left: Math.floor((m.width - width) / 2), top: Math.floor((m.height - height) / 2), width, height }));
}
