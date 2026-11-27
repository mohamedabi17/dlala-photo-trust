// Shared by the photo-check evaluation scripts (plan 40, phase 1): where files go, the seeded
// order every selection uses, hashing, and a polite downloader that resumes and verifies.
// Images never go into git: they live in ARTIFACTS, next to the repository, not inside it.
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

export const HERE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
/** <repo>/artifacts/eval-photos by default (git-ignored; overridable for another machine). */
export const ARTIFACTS = process.env.EVAL_PHOTOS_DIR || path.resolve(HERE, '..', 'artifacts', 'eval-photos');
/** Generic on purpose: no personal name or address leaves this machine. */
export const UA = 'DlalaEval/0.1 (photo-check evaluation)';
/** One seed for every selection, so a rebuild picks the same images from the same sources. */
export const SEED = 'dlala-photos-2026-11-24';
/** At most this many downloads at a time (brief of 24 Nov 2026). */
export const MAX_PARALLEL = 4;
/**
 * Wikimedia limits a client whose User-Agent carries no contact details to 10 requests a minute,
 * across all its sites, API and files together (mw:Wikimedia_APIs/Rate_limits, read 24 Nov 2026).
 * Every Wikimedia request therefore takes one shared turn, one at a time, 9 s apart (~6 a minute: 6.5 s still drew 429s after earlier bursts).
 */
const WIKIMEDIA = /(^|\.)(wikimedia|wikipedia|mediawiki|wikidata)\.org$/;
const WIKIMEDIA_GAP_MS = 9000;
const slotOf = (host) => (WIKIMEDIA.test(host) ? { key: 'wikimedia', parallel: 1, gapMs: WIKIMEDIA_GAP_MS } : { key: host, parallel: MAX_PARALLEL, gapMs: 0 });

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');
export const sha256File = (file) => sha256(fs.readFileSync(file));
/** A stable rank for `key` under `salt`: sorting by it is a seeded shuffle that does not move when items are added. */
export const rank = (salt, key) => crypto.createHash('sha256').update(`${SEED}:${salt}:${key}`).digest('hex');
export const seededOrder = (items, salt, key = (x) => x) => items.map((x) => [rank(salt, key(x)), x])
  .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(([, x]) => x);
/** Deterministic numbers in [0, 1) for one item (positions, phrases…), from its own hash. */
export function unitRandoms(salt, key, n) {
  const h = rank(salt, key);
  return Array.from({ length: n }, (_, i) => parseInt(h.slice(i * 8, i * 8 + 8), 16) / 2 ** 32);
}

/** JSON with one array item per line: readable diffs, small files. */
export function writeJsonLines(file, head, key, items) {
  const body = items.map((x) => `  ${JSON.stringify(x)}`).join(',\n');
  const headText = JSON.stringify(head, null, 2).replace(/\n}$/, '');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${headText},\n  ${JSON.stringify(key)}: [\n${body}\n  ]\n}\n`);
}
export const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));

/** Runs `fn` over `items`, at most `limit` at a time; returns results in input order. */
export async function pool(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}

// Per-host slots inside the global pool: a host never gets more than its share, nor requests
// closer than its gap (all Wikimedia hosts share one slot). A slot is held through the retries,
// so a 429 pauses every request to that host.
const busy = new Map();
const lastStart = new Map();
export async function hostSlot(host) {
  const { key, parallel, gapMs } = slotOf(host);
  for (;;) {
    const wait = (lastStart.get(key) || 0) + gapMs - Date.now();
    if ((busy.get(key) || 0) < parallel && wait <= 0) break;
    await sleep(Math.max(50, wait));
  }
  busy.set(key, (busy.get(key) || 0) + 1);
  lastStart.set(key, Date.now());
  return () => busy.set(key, busy.get(key) - 1);
}

/** GET with the generic User-Agent; retries 429 and 5xx, honouring Retry-After. Throws on other errors. */
export async function fetchOk(url, { timeoutMs = 60000, tries = 6 } = {}) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA }, signal: AbortSignal.timeout(timeoutMs) });
      if (r.ok) return r;
      last = new Error(`HTTP ${r.status} ${url.slice(0, 140)}`);
      if (r.status !== 429 && r.status < 500) { last.fatal = true; throw last; }
      // Back off 5 s, 10 s, 20 s… unless the server asks for longer.
      const seconds = Math.max(Number(r.headers.get('retry-after')) || 0, 5 * 2 ** i);
      console.warn(`${r.status} from ${new URL(url).host}: waiting ${seconds} s`);
      await sleep(seconds * 1000);
    } catch (e) {
      if (e.fatal) throw e;
      last = e;
      await sleep(3000 * (i + 1));
    }
  }
  throw last || new Error(`gave up: ${url.slice(0, 140)}`);
}

/**
 * Makes `dest` hold the file at `url`. An existing file is kept when its sha256 matches `expected`
 * (or, with no expected hash, as it is: that is a resumed run). Downloads go to `.part` and are
 * renamed only when complete, so an interrupted run never leaves a half file behind.
 * Returns { sha256, bytes, status: 'kept' | 'downloaded' } or throws (status 'mismatch' on a bad hash).
 */
export async function download(url, dest, expected) {
  if (fs.existsSync(dest)) {
    const have = sha256File(dest);
    if (!expected || have === expected) return { sha256: have, bytes: fs.statSync(dest).size, status: 'kept' };
    fs.unlinkSync(dest); // wrong content: fetch it again, then compare
  }
  const release = await hostSlot(new URL(url).host);
  try {
    const buf = Buffer.from(await (await fetchOk(url)).arrayBuffer());
    const got = sha256(buf);
    if (expected && got !== expected) throw Object.assign(new Error(`sha256 mismatch for ${url}: expected ${expected}, got ${got}`), { mismatch: true });
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(`${dest}.part`, buf);
    fs.renameSync(`${dest}.part`, dest);
    return { sha256: got, bytes: buf.length, status: 'downloaded' };
  } finally { release(); }
}

/** Bytes used under a folder (files only). */
export function folderBytes(dir) {
  if (!fs.existsSync(dir)) return 0;
  let n = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    n += e.isDirectory() ? folderBytes(p) : fs.statSync(p).size;
  }
  return n;
}
export const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

/** The laptop had ~12 GB free (24 Nov 2026): below 5 GB the scripts stop rather than fill the disk. */
const MIN_FREE_BYTES = 5e9;
/** Throws when the disk holding ARTIFACTS has less than MIN_FREE_BYTES free. */
export function checkDisk() {
  fs.mkdirSync(ARTIFACTS, { recursive: true });
  const s = fs.statfsSync(ARTIFACTS);
  const free = s.bavail * s.bsize;
  if (free < MIN_FREE_BYTES) throw new Error(`only ${mb(free)} free on the disk of ${ARTIFACTS}: stopping (limit ${mb(MIN_FREE_BYTES)})`);
  return free;
}
