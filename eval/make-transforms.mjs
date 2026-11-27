#!/usr/bin/env node
// Transformed copies for the « empreintes » set (plan 40, phase 1): for every photo marked
// `transformed` in fingerprint-manifest.json (all of `measure`, 1 000 of `calibrate`), the nine
// copies of lib/transforms.mjs, written as files so the later fingerprint run reads exactly the same
// bytes. Index: <artifacts>/fingerprint/transforms.json (id, parent, type, half, file, sha256; the
// collage also lists its three filler photos). Resumes: a file whose sha256 matches the index is kept.
//   node eval/make-transforms.mjs
import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import { ARTIFACTS, HERE, checkDisk, folderBytes, mb, pool, readJson, sha256, sha256File, unitRandoms, writeJsonLines } from './lib/common.mjs';
import { TRANSFORMS } from './lib/transforms.mjs';

const INDEX = path.join(ARTIFACTS, 'fingerprint/transforms.json');
/** Photos processed at once: leaves the laptop usable (sharp also uses its own threads). */
const PARALLEL = Math.max(1, Math.min(4, os.cpus().length - 1));
/** How often the free disk space is checked, in photos. */
const DISK_CHECK_EVERY = 200;

const manifest = readJson(path.join(HERE, 'fingerprint-manifest.json'));
const original = (id) => path.join(ARTIFACTS, 'fingerprint/originals', `${id}.jpg`);
const byHalf = { calibrate: manifest.images.filter((x) => x.half === 'calibrate'), measure: manifest.images.filter((x) => x.half === 'measure') };
const previous = new Map(fs.existsSync(INDEX) ? readJson(INDEX).transforms.map((t) => [t.id, t.sha256]) : []);

/** Three other photos of the same half, and the parent's cell (0-3), from the parent's seeded hash. */
function collagePlan(img) {
  const pool = byHalf[img.half];
  const u = unitRandoms('collage', img.id, 4);
  const fillers = [];
  for (let k = 1; k <= 3; k++) {
    let j = Math.floor(u[k] * pool.length);
    while (pool[j].id === img.id || fillers.includes(pool[j].id)) j = (j + 1) % pool.length;
    fillers.push(pool[j].id);
  }
  return { position: Math.floor(u[0] * 4), fillers };
}

const parents = manifest.images.filter((x) => x.transformed);
checkDisk();
let done = 0;
let made = 0;
const rows = await pool(parents, PARALLEL, async (img) => {
  const buf = fs.readFileSync(original(img.id));
  if (sha256(buf) !== img.sha256) throw new Error(`${img.id}: original does not match the manifest (run download.mjs)`);
  const plan = collagePlan(img);
  const out = [];
  for (const t of TRANSFORMS) {
    const id = `${img.id}~${t.type}`;
    const file = `fingerprint/transforms/${t.type}/${img.id}.jpg`;
    const abs = path.join(ARTIFACTS, file);
    let hash = fs.existsSync(abs) && previous.get(id) ? sha256File(abs) : null;
    if (!hash || hash !== previous.get(id)) {
      const ctx = { id: img.id, position: plan.position, fillers: t.type === 'collage_2x2' ? plan.fillers.map((f) => fs.readFileSync(original(f))) : [] };
      const bytes = await t.apply(buf, ctx);
      fs.mkdirSync(path.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, bytes);
      hash = sha256(bytes);
      made++;
    }
    out.push({ id, parent: img.id, type: t.type, half: img.half, file, sha256: hash, ...(t.type === 'collage_2x2' ? { position: plan.position, fillers: plan.fillers } : {}) });
  }
  if (++done % DISK_CHECK_EVERY === 0) { checkDisk(); console.log(`${done}/${parents.length}`); }
  return out;
});

const transforms = rows.flat();
const count = (k) => Object.fromEntries(Object.entries(transforms.reduce((a, t) => ({ ...a, [t[k]]: (a[t[k]] || 0) + 1 }), {})).sort());
writeJsonLines(INDEX, {
  about: 'Transformed copies of the fingerprint set (make-transforms.mjs). Files: <artifacts>/<file>. Parent ids refer to fingerprint-manifest.json.',
  sharp: sharp.versions.sharp, libvips: sharp.versions.vips,
  counts: { total: transforms.length, byType: count('type'), byHalf: count('half') },
}, 'transforms', transforms);
console.log(`transforms.json: ${transforms.length} copies of ${parents.length} photos (${made} written this run); ${mb(folderBytes(path.join(ARTIFACTS, 'fingerprint/transforms')))}`);
