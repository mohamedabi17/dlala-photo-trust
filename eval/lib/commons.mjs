// Wikimedia Commons, politely: one request at a time in Wikimedia's shared turn, `maxlag` so
// the API can ask us to wait when its databases lag, and back-off on 429. Only files whose licence
// is public domain, CC0, CC BY or CC BY-SA are kept; the licence is recorded as Commons states it.
import { UA, hostSlot, sleep } from './common.mjs';

const API = 'https://commons.wikimedia.org/w/api.php';
/** Seconds of replication lag above which the API refuses us (Wikimedia's recommended value). */
const MAXLAG = 5;
/** Thumbnail width: a Wikimedia standard size (1280 px; 1600 px is refused since T414805), still above what the checks look at. */
export const THUMB_WIDTH = 1280;
/** Smallest side kept: below it the photo says little about the object. */
const MIN_SIDE = 400;
const PHOTO_MIME = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/tiff']);

/** One request in Wikimedia's shared turn (lib/common.mjs); retries maxlag, 429 and 5xx inside it. */
export async function api(params) {
  const url = `${API}?${new URLSearchParams({ format: 'json', formatversion: '2', maxlag: String(MAXLAG), ...params })}`;
  const release = await hostSlot(new URL(API).host);
  try {
    for (let i = 0; i < 8; i++) {
      let r;
      try { r = await fetch(url, { headers: { 'User-Agent': UA }, signal: AbortSignal.timeout(60000) }); } catch { await sleep(5000 * (i + 1)); continue; }
      if (r.status === 429 || r.status >= 500) {
        const seconds = Math.max(Number(r.headers.get('retry-after')) || 0, 15 * 2 ** i);
        console.warn(`${r.status} from the Commons API: waiting ${seconds} s`);
        await sleep(seconds * 1000);
        continue;
      }
      const text = await r.text();
      let j;
      try { j = JSON.parse(text); } catch { await sleep(15000 * (i + 1)); continue; } // a plain-text "too many requests" page
      if (j.error?.code === 'maxlag') { await sleep(Math.max(Number(r.headers.get('retry-after')) || 5, 5) * 1000); continue; }
      if (j.error) throw new Error(`Commons API: ${j.error.code} ${j.error.info}`);
      return j;
    }
    throw new Error(`Commons API gave up: ${url.slice(0, 160)}`);
  } finally { release(); }
}

/** File titles directly in a category (no subcategories), in Commons' sort order. */
export async function categoryFileTitles(category, max = 500) {
  const titles = [];
  let cont = {};
  do {
    const j = await api({ action: 'query', list: 'categorymembers', cmtitle: `Category:${category}`, cmtype: 'file', cmlimit: '200', ...cont });
    for (const m of j.query?.categorymembers || []) titles.push(m.title);
    cont = j.continue || null;
  } while (cont && titles.length < max);
  return titles.slice(0, max);
}

/** File titles from Commons' search, bitmaps only: for a kind no category gathers. */
export async function searchFileTitles(query, max = 100) {
  const j = await api({ action: 'query', list: 'search', srsearch: `${query} filetype:bitmap`, srnamespace: '6', srlimit: String(Math.min(max, 500)) });
  return (j.query?.search || []).map((m) => m.title);
}

const strip = (h) => String(h || '').replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#0?39;/g, "'").replace(/\s+/g, ' ').trim();
const meta = (ii, k) => ii.extmetadata?.[k]?.value;

/** Commons' licence code and short name → kept or not. Public domain, CC0, CC BY, CC BY-SA only. */
export function licenceOk(code, short) {
  const c = String(code || '').toLowerCase();
  const s = String(short || '').toLowerCase();
  const tokens = `${c} ${s}`.split(/[^a-z0-9]+/);
  if (tokens.includes('nc') || tokens.includes('nd') || tokens.includes('noncommercial')) return false;
  return /^(pd|cc0|cc-by-\d|cc-by-sa-\d|cc-zero)/.test(c) || /^(public domain|cc0|cc by(-sa)? \d)/.test(s);
}

/** imageinfo (URL, sha1, size, licence, author) for titles, 50 per request. */
export async function imageInfos(titles) {
  const out = [];
  for (let i = 0; i < titles.length; i += 50) {
    const j = await api({
      action: 'query', titles: titles.slice(i, i + 50).join('|'), prop: 'imageinfo',
      iiprop: 'url|sha1|extmetadata|size|mime', iiurlwidth: String(THUMB_WIDTH),
      iiextmetadatafilter: 'LicenseShortName|License|LicenseUrl|UsageTerms|Artist|Credit|Restrictions|AttributionRequired',
    });
    for (const p of j.query?.pages || []) {
      const ii = p.imageinfo?.[0];
      if (!ii) continue;
      out.push({
        pageid: p.pageid, title: p.title, page: ii.descriptionurl, original: ii.url, url: String(ii.thumburl || ii.url).split('?')[0], // the tracking query is not part of the file
        width: ii.width, height: ii.height, mime: ii.mime, sourceSha1: ii.sha1,
        license: strip(meta(ii, 'LicenseShortName')) || strip(meta(ii, 'UsageTerms')), licenseCode: meta(ii, 'License') || null,
        licenseUrl: meta(ii, 'LicenseUrl') || null, author: strip(meta(ii, 'Artist')) || strip(meta(ii, 'Credit')) || null,
        restrictions: meta(ii, 'Restrictions') || null,
      });
    }
  }
  return out;
}

/** Why a file is not usable as a photo for the sets (null when it is). */
export function unusable(f) {
  if (!licenceOk(f.licenseCode, f.license)) return `licence: ${f.license || f.licenseCode || 'none'}`;
  if (!PHOTO_MIME.has(f.mime)) return `type: ${f.mime}`;
  if (Math.min(f.width, f.height) < MIN_SIDE) return `small: ${f.width}x${f.height}`;
  return null;
}
