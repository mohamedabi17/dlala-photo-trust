"""The vision service: a private HTTP server on the internal Docker network (plan phase 3).

Standard-library ThreadingHTTPServer: no web framework, less to install and less memory.
Request format and routes: vision/README.md. Every failure is a JSON error with a status code;
nothing a request sends can stop the process. Nothing of a request (pixels, text, boxes) is logged.
"""
import json
import logging
import signal
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import settings
from .fingerprint import fingerprint
from .images import BadInput, decode
from .models import ModelUnavailable, Registry, trim_memory
from .redact import redact
from .screen import screen
from .verify import verify

log = logging.getLogger("vision")
FRAMES_TYPE = "application/x-dlala-frames"
# The JSON header of a framed request is small (boxes, options): refuse anything larger.
MAX_META_BYTES = 1 << 20
FRAME_LENGTH_BYTES = 4
STARTED = time.time()
ROUTES = {"/fingerprint": 1, "/screen": 1, "/redact": 1, "/verify": 2}
# The idle watcher looks at least this often.
IDLE_CHECK_MAX_S = 30

def proc_memory_mb():
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    out[line.split(":")[0]] = round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return {"rss_mb": out.get("VmRSS"), "peak_rss_mb": out.get("VmHWM")}


def parse_body(content_type, body, expected_parts):
    """(meta dict, [image bytes]) from a framed body or a bare image body."""
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype == FRAMES_TYPE:
        if len(body) < FRAME_LENGTH_BYTES:
            raise BadInput(400, "bad_frames", "a framed body starts with 4 bytes giving the JSON length")
        n = struct.unpack(">I", body[:FRAME_LENGTH_BYTES])[0]
        if n > MAX_META_BYTES or FRAME_LENGTH_BYTES + n > len(body):
            raise BadInput(400, "bad_frames", "the JSON length is larger than the body or the limit")
        try:
            meta = json.loads(body[FRAME_LENGTH_BYTES:FRAME_LENGTH_BYTES + n].decode("utf-8")) if n else {}
        except (UnicodeDecodeError, ValueError):
            raise BadInput(400, "bad_meta", "the JSON header is not valid JSON")
        if not isinstance(meta, dict):
            raise BadInput(400, "bad_meta", "the JSON header must be an object")
        rest = body[FRAME_LENGTH_BYTES + n:]
        lengths = meta.pop("lengths", [len(rest)])
        if not isinstance(lengths, list) or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in lengths) or sum(lengths) != len(rest):
            raise BadInput(400, "bad_frames", "meta.lengths must list the byte length of each image, adding up to the rest of the body")
        parts, pos = [], 0
        for x in lengths:
            parts.append(rest[pos:pos + x])
            pos += x
    elif ctype.startswith("image/") or ctype == "application/octet-stream":
        meta, parts = {}, [body]
    else:
        raise BadInput(415, "bad_content_type", f"send {FRAMES_TYPE}, image/* or application/octet-stream")
    if len(parts) != expected_parts:
        raise BadInput(400, "bad_frames", f"this route takes {expected_parts} image(s), got {len(parts)}")
    return meta, parts


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = settings.SOCKET_TIMEOUT_S
    server_version = "dlala-vision"
    sys_version = ""

    def log_message(self, fmt, *args):  # the access log is ours (below), without request content
        pass

    def _send(self, status, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        return status

    def _error(self, status, code, message):
        return self._send(status, {"error": {"code": code, "message": message}})

    def do_GET(self):
        t = time.perf_counter()
        if self.path.split("?")[0] != "/health":
            status = self._error(404, "not_found", "unknown route")
        else:
            reg = self.server.registry
            models = reg.status()
            required = ("siglip", "dino", "yunet", "prompts")
            ok = all(models[k] in ("present", "loaded") for k in required)
            status = self._send(200 if ok else 503, {
                "status": "ok" if ok else "degraded", "api_version": settings.API_VERSION,
                "dino_variant": reg.dino_variant, "models": models,
                "errors": reg.errors(), "uptime_s": round(time.time() - STARTED),
                "idle_unload_s": self.server.idle_unload_s, "unloads": self.server.unloads, **proc_memory_mb(),
            })
        self._access(status, t)

    def do_POST(self):
        t = time.perf_counter()
        status = 500
        try:
            status = self._post()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
            status = 499
        except Exception:  # last resort: a bug answers 500, the server keeps running
            log.exception("unexpected error on %s", self.path)
            self.close_connection = True
            try:
                status = self._error(500, "internal", "unexpected error (logged)")
            except OSError:
                pass
        finally:
            trim_memory()
            self._access(status, t)

    def _post(self):
        route = self.path.split("?")[0]
        if route not in ROUTES:
            self.close_connection = True
            return self._error(404, "not_found", "unknown route")
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            return self._error(411, "length_required", "send Content-Length (no chunked bodies)")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            return self._error(411, "length_required", "Content-Length is required")
        if length <= 0:
            return self._error(400, "empty_body", "the request has no body")
        if length > settings.MAX_BODY_BYTES:
            self.close_connection = True  # the body is not read: the connection cannot be reused
            return self._error(413, "body_too_large", f"the body is larger than {settings.MAX_BODY_BYTES} bytes")
        srv = self.server
        with srv.waiting_lock:
            if srv.waiting >= srv.max_waiting:
                self.close_connection = True  # the body is not read
                return self._error(503, "busy", f"{srv.waiting} requests are already waiting; try again")
            srv.waiting += 1
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                self.close_connection = True
                return self._error(400, "short_body", "the body is shorter than Content-Length")
            got_slot = srv.slots.acquire(timeout=settings.QUEUE_WAIT_S)
        finally:
            with srv.waiting_lock:
                srv.waiting -= 1
        if not got_slot:
            return self._error(503, "busy", "all slots are busy; try again")
        srv.last_activity = time.monotonic()
        try:
            meta, parts = parse_body(self.headers.get("Content-Type"), body, ROUTES[route])
            del body
            images = [decode(p) for p in parts]
            reg = self.server.registry
            started = time.perf_counter()
            if route == "/fingerprint":
                out = fingerprint(reg, images[0], meta)
            elif route == "/screen":
                out = screen(reg, images[0], meta)
            elif route == "/verify":
                out = verify(images[0], images[1], meta)
            else:
                jpeg, boxes = redact(images[0], meta)
                return self._send(200, jpeg, "image/jpeg", {"X-Redacted-Boxes": json.dumps(boxes, separators=(",", ":"))})
            out["took_ms"] = round((time.perf_counter() - started) * 1000)
            return self._send(200, out)
        except BadInput as e:
            return self._error(e.status, e.code, e.message)
        except ModelUnavailable as e:
            return self._error(503, "model_unavailable", str(e))
        finally:
            srv.last_activity = time.monotonic()
            srv.slots.release()

    def _access(self, status, t):
        log.info("%s %s %s %dms", self.command, self.path.split("?")[0], status, (time.perf_counter() - t) * 1000)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, registry, idle_unload_s=None):
        super().__init__(addr, Handler)
        self.registry = registry
        self.slots = threading.BoundedSemaphore(settings.MAX_INFLIGHT)
        self.waiting = 0
        self.max_waiting = settings.MAX_WAITING
        self.waiting_lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.idle_unload_s = settings.IDLE_UNLOAD_S if idle_unload_s is None else idle_unload_s
        self.unloads = 0
        if self.idle_unload_s > 0:
            threading.Thread(target=self._idle_watch, name="idle-unload", daemon=True).start()

    def _idle_watch(self):
        """Unload the models after idle_unload_s without a request, while no request holds a slot."""
        # Check often enough to unload within a quarter of the delay (at least every second).
        period = max(1.0, min(IDLE_CHECK_MAX_S, self.idle_unload_s / 4))
        while True:
            time.sleep(period)
            if time.monotonic() - self.last_activity < self.idle_unload_s or not self.registry.loaded_models():
                continue
            taken = 0
            while taken < settings.MAX_INFLIGHT and self.slots.acquire(blocking=False):
                taken += 1
            try:
                if taken == settings.MAX_INFLIGHT and self.waiting == 0:
                    if self.registry.unload():
                        self.unloads += 1
            finally:
                for _ in range(taken):
                    self.slots.release()


def make_server(host=None, port=None, registry=None, idle_unload_s=None):
    return Server((host or settings.HOST, settings.PORT if port is None else port), registry or Registry(), idle_unload_s)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    srv = make_server()
    log.info("vision %s listening on %s:%s (dino %s, models in %s; loaded on first use)",
             settings.API_VERSION, settings.HOST, settings.PORT, srv.registry.dino_variant, settings.MODELS_DIR)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
