"""Test images: listed in fixtures.json (URL, sha256, licence), downloaded once into a cache.

The cache is VISION_TEST_CACHE (default ~/.cache/dlala-vision-tests). A file whose sha256 does not
match is downloaded again; a download that still does not match fails the tests.
"""
import hashlib
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Wikimedia asks automated clients for a descriptive User-Agent and throttles fast ones.
USER_AGENT = "dlala-vision-tests/1.0 (test fixtures)"
PAUSE_BETWEEN_DOWNLOADS_S = 2
TIMEOUT_S = 120


def load_fixtures():
    with open(os.path.join(HERE, "fixtures.json")) as f:
        return json.load(f)["files"]


def default_cache():
    return os.environ.get("VISION_TEST_CACHE", os.path.expanduser("~/.cache/dlala-vision-tests"))


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fixture_path(entry, cache=None):
    cache = cache or default_cache()
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, entry["id"] + entry["ext"])
    if os.path.isfile(path) and _sha256(path) == entry["sha256"]:
        return path
    req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        data = r.read()
    time.sleep(PAUSE_BETWEEN_DOWNLOADS_S)
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise AssertionError(f"fixture {entry['id']} from {entry['url']} does not match its sha256")
    with open(path, "wb") as f:
        f.write(data)
    return path


def fixture_by_id(fid, cache=None):
    for e in load_fixtures():
        if e["id"] == fid:
            return fixture_path(e, cache)
    raise KeyError(fid)
