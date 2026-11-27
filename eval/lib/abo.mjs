// Amazon Berkeley Objects (ABO, CC BY 4.0, credit: Amazon.com): catalogue photos of ordinary
// products. Only the image metadata (6 MB) and single small files are fetched, never the tar
// archives. Layout checked on 24 Nov 2026 in images/README.md of the bucket: `path` in
// images/metadata/images.csv.gz is relative to images/small/ (largest side at most 256 px).
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { ARTIFACTS, download, seededOrder } from './common.mjs';

const BUCKET = 'https://amazon-berkeley-objects.s3.amazonaws.com';
export const ABO_META_URL = `${BUCKET}/images/metadata/images.csv.gz`;
export const ABO_LICENSE = 'CC BY 4.0';
export const ABO_AUTHOR = 'Amazon.com (Amazon Berkeley Objects dataset)';
export const ABO_PAGE = `${BUCKET}/index.html`;
export const smallUrl = (p) => `${BUCKET}/images/small/${p}`;

/** Disjoint draws from one seeded order of ABO: the ordinary photos first, then the fingerprint set. */
export const ORDINARY_COUNT = 1000;
export const FINGERPRINT_COUNT = 10000;
/**
 * Originals kept: both sides at least 256 px, so the small file is a full 256 px photo, not a
 * thumbnail of a banner; aspect ratio at most 2, which drops strips and size charts.
 */
const MIN_ORIGINAL_SIDE = 256;
const MAX_ASPECT = 2;

/** The metadata, downloaded once into the cache; its sha256 is recorded in each manifest. */
export async function aboMetadata() {
  const file = path.join(ARTIFACTS, 'cache/abo-images.csv.gz');
  const { sha256 } = await download(ABO_META_URL, file);
  const rows = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8').trim().split(/\r?\n/).slice(1).map((line) => {
    const [imageId, height, width, p] = line.split(',');
    return { imageId, height: Number(height), width: Number(width), path: p };
  });
  return { sha256, rows };
}

/** id from the file's own hex name (unique in ABO), safe in file names. */
export const aboId = (row) => `abo-${path.basename(row.path, path.extname(row.path))}`;

/** The two disjoint ABO draws, in seeded order. */
export function aboDraws(rows) {
  const usable = rows.filter((r) => r.path.endsWith('.jpg') && Math.min(r.width, r.height) >= MIN_ORIGINAL_SIDE
    && Math.max(r.width, r.height) / Math.min(r.width, r.height) <= MAX_ASPECT);
  const order = seededOrder(usable, 'abo', (r) => r.path);
  return { usable: usable.length, ordinary: order.slice(0, ORDINARY_COUNT), fingerprint: order.slice(ORDINARY_COUNT, ORDINARY_COUNT + FINGERPRINT_COUNT) };
}
