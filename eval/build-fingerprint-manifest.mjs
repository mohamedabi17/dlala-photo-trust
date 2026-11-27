#!/usr/bin/env node
// The « empreintes » set (plan 40, phase 1): 10 000 ABO photos, disjoint from the 1 000 ordinary
// photos of the « objets » set (both come from one seeded order of ABO, see lib/abo.mjs), split by
// a seeded shuffle into a `calibrate` half and a `measure` half: thresholds are chosen on one and
// reported on the other, never both on the same images. Every `measure` photo is transformed, and
// 1 000 `calibrate` photos (make-transforms.mjs). Downloads each original once into the artifacts
// folder and pins its sha256 in fingerprint-manifest.json.
//   node eval/build-fingerprint-manifest.mjs
import path from 'path';
import { ARTIFACTS, HERE, MAX_PARALLEL, SEED, download, pool, seededOrder, writeJsonLines } from './lib/common.mjs';
import { ABO_AUTHOR, ABO_LICENSE, ABO_META_URL, ABO_PAGE, aboDraws, aboId, aboMetadata, smallUrl } from './lib/abo.mjs';
import { TRANSFORMS } from './lib/transforms.mjs';

/** Calibration photos that also get transformed copies (brief of 24 Nov 2026): enough to fit a threshold. */
const CALIBRATE_TRANSFORMED = 1000;

const { sha256: metaSha, rows } = await aboMetadata();
const { usable, fingerprint } = aboDraws(rows);
const half = Math.floor(fingerprint.length / 2);
const split = seededOrder(fingerprint, 'split', (r) => r.path);
const calibrate = new Set(split.slice(0, half).map((r) => r.path));
const transformedCalibrate = new Set(seededOrder(split.slice(0, half), 'calibrate-transformed', (r) => r.path).slice(0, CALIBRATE_TRANSFORMED).map((r) => r.path));

let done = 0;
const images = await pool(split, MAX_PARALLEL, async (r) => {
  const id = aboId(r);
  const { sha256 } = await download(smallUrl(r.path), path.join(ARTIFACTS, 'fingerprint/originals', `${id}.jpg`));
  if (++done % 1000 === 0) console.log(`${done}/${split.length}`);
  const inCalibrate = calibrate.has(r.path);
  return { id, path: r.path, sha256, half: inCalibrate ? 'calibrate' : 'measure', transformed: !inCalibrate || transformedCalibrate.has(r.path) };
});
// Stored in id order (not split order), so the file reads the same however the split moves.
images.sort((a, b) => (a.id < b.id ? -1 : 1));

writeJsonLines(path.join(HERE, 'fingerprint-manifest.json'), {
  about: 'Fingerprint set (plan 40, phase 1): ABO small photos, split into calibrate and measure halves. Files: <artifacts>/fingerprint/originals/<id>.jpg; transformed copies are listed in <artifacts>/fingerprint/transforms.json (make-transforms.mjs).',
  source: { name: 'Amazon Berkeley Objects', license: ABO_LICENSE, author: ABO_AUTHOR, page: ABO_PAGE, metadata: ABO_META_URL, metadataSha256: metaSha, fileUrl: smallUrl('<path>') },
  seed: SEED,
  counts: { aboUsable: usable, images: images.length, calibrate: images.filter((x) => x.half === 'calibrate').length, measure: images.filter((x) => x.half === 'measure').length, transformed: images.filter((x) => x.transformed).length },
  transforms: TRANSFORMS.map(({ type, params, about }) => ({ type, params, about })),
}, 'images', images);
console.log(`fingerprint-manifest.json: ${images.length} photos (${images.filter((x) => x.transformed).length} to transform)`);
