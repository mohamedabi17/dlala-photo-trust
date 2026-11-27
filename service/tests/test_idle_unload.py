"""Idle unload: after VISION_IDLE_UNLOAD_S without a request the models leave memory, and the
next request loads them again (here with a 2-second delay instead of 600)."""
import time
import unittest

from .common import ServerCase, load

# SigLIP 2 alone holds about 340 MB once loaded: at least this much must go back.
MIN_RSS_DROP_MB = 200
WAIT_FOR_UNLOAD_S = 30


class IdleUnload(ServerCase):
    idle_unload_s = 2

    def health(self):
        status, out, _ = self.request("GET", "/health")
        return out

    def test_unload_then_reload(self):
        photo = load("wm-33284915")
        status, out, _ = self.post("/screen", {"parts": ["siglip", "faces", "ocr"]}, photo)
        self.assertEqual(status, 200, out)
        loaded = self.health()
        self.assertEqual(loaded["models"]["siglip"], "loaded")
        deadline = time.monotonic() + WAIT_FOR_UNLOAD_S
        # The models leave memory before the unload is counted (gc and malloc_trim run in between): wait for both.
        while (self.health()["models"]["siglip"] == "loaded" or self.health()["unloads"] < 1) and time.monotonic() < deadline:
            time.sleep(0.5)
        idle = self.health()
        drop = loaded["rss_mb"] - idle["rss_mb"]
        print(f"  idle unload: rss {loaded['rss_mb']} MB loaded, {idle['rss_mb']} MB after unloading ({drop:.0f} MB back); models {idle['models']}")
        self.assertEqual(idle["models"]["siglip"], "present")
        self.assertGreaterEqual(idle["unloads"], 1)
        self.assertGreaterEqual(drop, MIN_RSS_DROP_MB)
        status, out, _ = self.post("/screen", {"parts": ["siglip"]}, photo)
        self.assertEqual(status, 200, out)
        self.assertEqual(max(out["siglip"], key=out["siglip"].get), "ivory")


if __name__ == "__main__":
    unittest.main()
