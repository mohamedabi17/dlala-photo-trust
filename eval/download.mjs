#!/usr/bin/env node
// Fetches the files of both manifests into the artifacts folder (never into git) and checks every
// sha256. Safe to stop and run again: a file already there with the right hash is kept, a missing
// one is fetched, a wrong one is fetched again and must then match. At most 4 downloads at a time
// (2 on Wikimedia's file servers), with a generic User-Agent. Files made by a script (injection
// images, transformed copies) are only checked: the script that makes them is named when one is
// missing. Exits 1 when anything is missing or does not match.
//   node eval/download.mjs                 (everything)
//   node eval/download.mjs vision          (only the « objets » set)
//   node eval/download.mjs fingerprint     (only the « empreintes » set)
import fs from 'fs';
import path from 'path';
import { ARTIFACTS, HERE, MAX_PARALLEL, checkDisk, download, folderBytes, mb, pool, readJson, sha256File } from './lib/common.mjs';
import { ABO_META_URL, smallUrl } from './lib/abo.mjs';

const only = process.argv[2];
if (only && !['vision', 'fingerprint'].includes(only)) { console.error('usage: download.mjs [vision|fingerprint]'); process.exit(2); }
/** Free disk space is checked every this many files. */
const DISK_CHECK_EVERY = 500;

const jobs = [];
const generated = [];
if (!only || only === 'vision') {
  for (const e of readJson(path.join(HERE, 'vision-manifest.json')).images) {
    if (e.url) jobs.push({ url: e.url, file: e.file, sha256: e.sha256 });
    else generated.push({ file: e.file, sha256: e.sha256, by: 'make-injection.mjs' });
  }
}
if (!only || only === 'fingerprint') {
  const fp = readJson(path.join(HERE, 'fingerprint-manifest.json'));
  jobs.push({ url: ABO_META_URL, file: 'cache/abo-images.csv.gz', sha256: fp.source.metadataSha256 });
  for (const x of fp.images) jobs.push({ url: smallUrl(x.path), file: `fingerprint/originals/${x.id}.jpg`, sha256: x.sha256 });
  const index = path.join(ARTIFACTS, 'fingerprint/transforms.json');
  if (fs.existsSync(index)) for (const t of readJson(index).transforms) generated.push({ file: t.file, sha256: t.sha256, by: 'make-transforms.mjs' });
  else console.log('no fingerprint/transforms.json yet: run make-transforms.mjs after this');
}

console.log(`free disk: ${mb(checkDisk())}; ${jobs.length} files to fetch or check, ${generated.length} generated files to check`);
const tally = { kept: 0, downloaded: 0, failed: 0 };
const failures = [];
let n = 0;
await pool(jobs, MAX_PARALLEL, async (j) => {
  try { tally[(await download(j.url, path.join(ARTIFACTS, j.file), j.sha256)).status]++; } catch (e) { tally.failed++; failures.push(`${j.file}: ${e.message}`); }
  if (++n % DISK_CHECK_EVERY === 0) { checkDisk(); console.log(`${n}/${jobs.length}`); }
});
const missing = { 'make-injection.mjs': 0, 'make-transforms.mjs': 0 };
let bad = 0;
for (const g of generated) {
  const abs = path.join(ARTIFACTS, g.file);
  if (!fs.existsSync(abs)) missing[g.by]++;
  else if (sha256File(abs) !== g.sha256) { bad++; failures.push(`${g.file}: sha256 differs from the manifest (regenerate with ${g.by})`); }
}

console.log(`fetched: ${tally.downloaded}, already there: ${tally.kept}, failed: ${tally.failed}; generated files: ${generated.length - bad - missing['make-injection.mjs'] - missing['make-transforms.mjs']} ok, ${bad} differ`);
for (const [by, k] of Object.entries(missing)) if (k) console.log(`${k} generated files missing: run ${by}`);
if (failures.length) console.log(failures.slice(0, 30).join('\n') + (failures.length > 30 ? `\n… and ${failures.length - 30} more` : ''));
console.log(`artifacts folder: ${mb(folderBytes(ARTIFACTS))}`);
process.exit(failures.length || Object.values(missing).some(Boolean) ? 1 : 0);
