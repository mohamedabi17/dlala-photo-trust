"""Download the model files listed in models.lock.json, once, and check each one.

Run by the deployment's fetch script (deploy/fetch-models.sh in Dlala) inside a short-lived container of the vision image (the only
time a vision container reaches the internet). Also usable on a work computer:

    python -m app.fetch_models --lock models.lock.json --dest <models dir> [--groups vision vlm reference]

Each file is written to <name>.part, hashed while it downloads, and renamed only when its
size and sha256 match the lock. A file already in place is re-hashed, never trusted blindly.
Exit code 0 when every requested file is present and correct.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

# Groups fetched when none are named: what production runs (DINO_VARIANT=small).
DEFAULT_GROUPS = ("vision", "vlm")
# Read and hash in blocks of 1 MiB: small enough for a 1-core server, large enough to be fast.
CHUNK_BYTES = 1 << 20
# Keep this much free on the disk after the downloads (same rule as deploy/release.sh MIN_FREE_GB).
DEFAULT_MIN_FREE_GB = 5
# Network: give up on a stalled connection after this many seconds, retry a failed file this many times.
SOCKET_TIMEOUT_S = 60
DOWNLOAD_ATTEMPTS = 3
RETRY_PAUSE_S = 5
# Print progress every this many percent of a large file.
PROGRESS_STEP_PERCENT = 10
GIB = 1 << 30
USER_AGENT = "dlala-fetch-models/1"


def human(n):
    """Bytes as a short text: 94.6 MB, 1.1 GB."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.0f} kB"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK_BYTES)
            if not block:
                return h.hexdigest()
            h.update(block)


def is_good(path, entry):
    return os.path.isfile(path) and os.path.getsize(path) == entry["size"] and sha256_of(path) == entry["sha256"]


def download(entry, dest):
    """Download one file to dest (via dest.part, resumed if a part is there). True when it checks out."""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > entry["size"]:
                os.remove(part)
                have = 0
            h = hashlib.sha256()
            if have:
                with open(part, "rb") as f:
                    for block in iter(lambda: f.read(CHUNK_BYTES), b""):
                        h.update(block)
            req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT_S) as r:
                if not r.geturl().startswith("https://"):
                    print("    refused: the download was redirected to a non-https address", flush=True)
                    return False
                if have and r.status != 206:
                    # The server ignored the range: start again from zero.
                    have = 0
                    h = hashlib.sha256()
                mode = "ab" if have else "wb"
                done, next_mark = have, PROGRESS_STEP_PERCENT
                with open(part, mode) as f:
                    while True:
                        block = r.read(CHUNK_BYTES)
                        if not block:
                            break
                        f.write(block)
                        h.update(block)
                        done += len(block)
                        pct = done * 100 // entry["size"]
                        if entry["size"] > 50e6 and pct >= next_mark:
                            print(f"    {pct}% of {human(entry['size'])}", flush=True)
                            next_mark = pct - pct % PROGRESS_STEP_PERCENT + PROGRESS_STEP_PERCENT
            if done != entry["size"] or h.hexdigest() != entry["sha256"]:
                os.remove(part)
                print(f"    the downloaded file is not the expected one (size or checksum); attempt {attempt} of {DOWNLOAD_ATTEMPTS}", flush=True)
                continue
            os.replace(part, dest)
            os.chmod(dest, 0o444)
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"    download interrupted ({type(e).__name__}: {e}); attempt {attempt} of {DOWNLOAD_ATTEMPTS}", flush=True)
            time.sleep(RETRY_PAUSE_S)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.lock.json"))
    ap.add_argument("--dest", default=os.environ.get("MODELS_DIR", "/models"))
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--check", action="store_true", help="only check what is there; download nothing")
    ap.add_argument("--min-free-gb", type=float, default=float(os.environ.get("MIN_FREE_GB", DEFAULT_MIN_FREE_GB)))
    a = ap.parse_args(argv)

    with open(a.lock) as f:
        lock = json.load(f)
    wanted = [e for e in lock["files"] if e["group"] in a.groups]
    if not wanted:
        print(f"No model file belongs to the groups {a.groups}.")
        return 2
    plain = [e["path"] for e in wanted if not e["url"].startswith("https://")]
    if plain:
        print(f"Refused: the lock gives a non-https address for {', '.join(plain)}. Nothing downloaded.")
        return 2
    print(f"Model files: {len(wanted)} ({', '.join(a.groups)}), {human(sum(e['size'] for e in wanted))} in total, in {a.dest}", flush=True)

    missing = []
    for e in wanted:
        path = os.path.join(a.dest, e["path"])
        if is_good(path, e):
            print(f"  ok      {e['path']} (already there, checked)", flush=True)
        else:
            missing.append(e)
            print(f"  {'MISSING' if a.check else 'to get '} {e['path']} ({human(e['size'])}, licence {e['licence']})", flush=True)
    if a.check:
        if missing:
            print(f"{len(missing)} file(s) missing or damaged: run app.fetch_models again.")
            return 1
        print("All model files are present and correct.")
        return 0
    if not missing:
        print("All model files are present and correct. Nothing to download.")
        return 0

    os.makedirs(a.dest, exist_ok=True)
    need = sum(e["size"] for e in missing)
    free = shutil.disk_usage(a.dest).free
    if free - need < a.min_free_gb * GIB:
        print(f"Not enough disk: {human(free)} free, the downloads need {human(need)} and {a.min_free_gb:g} GB must stay free. Nothing downloaded.")
        return 1

    failed = []
    for e in missing:
        path = os.path.join(a.dest, e["path"])
        if os.path.exists(path):
            os.chmod(path, 0o644)
            os.remove(path)  # wrong size or checksum: never keep it
        print(f"  getting {e['path']} ({human(e['size'])}) from {e['source_page']}", flush=True)
        if download(e, path):
            print(f"  ok      {e['path']} (downloaded, checksum correct)", flush=True)
        else:
            failed.append(e["path"])
            print(f"  FAILED  {e['path']}", flush=True)
    if failed:
        print(f"{len(failed)} file(s) could not be downloaded: {', '.join(failed)}. Run the script again later; finished files are kept.")
        return 1
    print(f"Done: all {len(wanted)} model files are present and correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
