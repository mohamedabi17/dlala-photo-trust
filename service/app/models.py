"""The models, each loaded on its first use and kept afterwards (memory budget: plan phase 0, point 8).

Every file is named by models.lock.json (path, size, sha256); a missing or damaged file makes
the routes that need it answer 503, never a result computed without it.
"""
import ctypes
import gc
import hashlib
import json
import logging
import os
import threading

import numpy as np
from PIL import Image

from . import settings
from .images import to_tensor

log = logging.getLogger("vision.models")

# SigLIP 2 base patch16 224, preprocessor_config.json: 224 x 224, bilinear (resample 2), mean and std 0.5.
SIGLIP_SIZE = 224
SIGLIP_RESAMPLE = Image.Resampling.BILINEAR
SIGLIP_MEAN = SIGLIP_STD = (0.5, 0.5, 0.5)
# DINOv2, preprocessor_config.json: bicubic (resample 3), ImageNet mean and std. The plan keeps the
# whole image squared to 224 (no centre crop), as in the benchmark (plan phase 3).
DINO_SIZE = 224
DINO_RESAMPLE = Image.Resampling.BICUBIC
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
# YuNet's own defaults (OpenCV FaceDetectorYN.create): NMS 0.3, at most 5000 faces. The score
# floor is lower than the default 0.9 on purpose: the API applies the calibrated threshold.
YUNET_NMS = 0.3
YUNET_TOP_K = 5000
YUNET_SCORE_FLOOR = 0.5
# RapidOCR: text detection on a copy whose longest side is at most OCR_DET_SIDE. Its default
# (shorter side at least 736) ran detection on 2000 x 1600 px: 11.9 s and +700 MB on a 10 MP photo
# here; at 1280, 3.7 s and +390 MB; at 960, 2.8 s and +300 MB (vision/README.md, "Measurements").
# Lines are read on the copy given to RapidOCR, at most OCR_MAX_SIDE. Defaults chosen for memory
# (review of 24/11): phase 1 must check the OCR's accuracy at these sizes and may raise them.
OCR_DET_SIDE = int(os.environ.get("OCR_DET_SIDE", "736"))
OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1280"))
# What unload() drops: the models. Prompt vectors (0.2 MB) and the probe stay.
UNLOADABLE = ("siglip", "dino-small", "dino-base", "yunet", "ocr")

# Full precision: dynamic int8 versions fell to a cosine of 0.70 (SigLIP 2) and 0.96 (DINOv2-S)
# against fp32, below the 0.99 the plan requires (vision/README.md, "int8").
LOCK_IDS = {"siglip": "siglip2-vision", "dino-small": "dinov2-small", "dino-base": "dinov2-base", "yunet": "yunet-2023mar"}


try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc: nothing to trim
    _libc = None


def trim_memory():
    """Give freed heap memory back to the system (glibc keeps it otherwise)."""
    if _libc is not None:
        _libc.malloc_trim(0)


_ort_ready = False


def ort_module():
    """onnxruntime, with its telemetry events turned off (they only exist on Windows builds,
    but the call is the documented switch: onnxruntime.disable_telemetry_events)."""
    global _ort_ready
    import onnxruntime as ort

    if not _ort_ready:
        ort.disable_telemetry_events()
        _ort_ready = True
    return ort


class ModelUnavailable(Exception):
    """A model file is missing, damaged or failed to load: the route answers 503."""


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Registry:
    def __init__(self, models_dir=None, dino_variant=None, lock_path=None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.dino_variant = dino_variant or settings.DINO_VARIANT
        if self.dino_variant not in ("small", "base"):
            raise ValueError("DINO_VARIANT must be small or base")
        with open(lock_path or settings.LOCK_PATH) as f:
            self.lock_entries = {e["id"]: e for e in json.load(f)["files"]}
        self._load_lock = threading.Lock()
        self._loaded = {}
        self._errors = {}
        # Stateful OpenCV and OCR objects are used by one request at a time.
        self.yunet_lock = threading.Lock()
        self.ocr_lock = threading.Lock()

    # Files -----------------------------------------------------------------------------------
    def entry(self, kind):
        return self.lock_entries[LOCK_IDS[kind]]

    def path(self, kind):
        return os.path.join(self.models_dir, self.entry(kind)["path"])

    def file_state(self, kind):
        e = self.entry(kind)
        p = os.path.join(self.models_dir, e["path"])
        if not os.path.isfile(p):
            return "missing"
        if os.path.getsize(p) != e["size"]:
            return "wrong_size"
        return "present"

    def _checked_path(self, kind):
        e = self.entry(kind)
        p = os.path.join(self.models_dir, e["path"])
        state = self.file_state(kind)
        if state != "present":
            raise ModelUnavailable(f"{e['path']} is {state}: fetch it with python -m app.fetch_models")
        if settings.VERIFY_SHA_ON_LOAD and sha256_of(p) != e["sha256"]:
            raise ModelUnavailable(f"{e['path']} does not match its sha256: fetch it again with python -m app.fetch_models")
        return p

    # Lazy loading ----------------------------------------------------------------------------
    def _get(self, name, loader):
        obj = self._loaded.get(name)
        if obj is not None:
            return obj
        with self._load_lock:
            obj = self._loaded.get(name)
            if obj is None:
                try:
                    obj = loader()
                except ModelUnavailable as e:
                    self._errors[name] = str(e)
                    raise
                except Exception as e:  # a model that does not load is unavailable, not a crash
                    log.exception("loading %s failed", name)
                    self._errors[name] = f"{type(e).__name__}: {e}"
                    raise ModelUnavailable(f"{name} failed to load")
                self._loaded[name] = obj
                self._errors.pop(name, None)
                # Loading parses the file into a second copy of the weights; give it back now
                # (SigLIP 2: 650 MB right after loading, 414 MB after this).
                trim_memory()
                log.info("loaded %s", name)
            return obj

    def _ort(self, kind):
        ort = ort_module()
        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.ORT_THREADS
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # No arena: memory goes back after each call instead of staying reserved (lower steady RSS).
        so.enable_cpu_mem_arena = False
        return ort.InferenceSession(self._checked_path(kind), sess_options=so, providers=["CPUExecutionProvider"])

    def siglip(self):
        return self._get("siglip", lambda: self._ort("siglip"))

    def dino(self):
        kind = f"dino-{self.dino_variant}"
        return self._get(kind, lambda: self._ort(kind))

    def yunet(self):
        import cv2

        def load():
            return cv2.FaceDetectorYN.create(self._checked_path("yunet"), "", (320, 320), YUNET_SCORE_FLOOR, YUNET_NMS, YUNET_TOP_K)

        return self._get("yunet", load)

    def ocr(self):
        def load():
            ort_module()  # RapidOCR builds its own ONNX Runtime sessions
            from rapidocr import RapidOCR

            return RapidOCR(params={
                "Global.log_level": "warning",
                "Global.max_side_len": OCR_MAX_SIDE,
                "Det.limit_type": "max",
                "Det.limit_side_len": OCR_DET_SIDE,
                "EngineConfig.onnxruntime.intra_op_num_threads": settings.ORT_THREADS,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })

        return self._get("ocr", load)

    def prompts(self):
        def load():
            meta_path = os.path.join(settings.PROMPTS_DIR, "prompts.json")
            vec_path = os.path.join(settings.PROMPTS_DIR, "prompts.npy")
            if not (os.path.isfile(meta_path) and os.path.isfile(vec_path)):
                raise ModelUnavailable("prompts.json / prompts.npy are missing: run tools/build-prompts.py")
            with open(meta_path) as f:
                meta = json.load(f)
            vecs = l2(np.load(vec_path).astype(np.float32))
            if vecs.shape[0] != len(meta["prompts"]):
                raise ModelUnavailable("prompts.npy and prompts.json do not match")
            return {"meta": meta, "vectors": vecs, "groups": [p["group"] for p in meta["prompts"]],
                    "scale": float(np.exp(meta["logit_scale"])), "bias": float(meta["logit_bias"])}

        return self._get("prompts", load)

    def probe(self):
        """The logistic probe of phase 1, or None while it does not exist (or is unusable)."""
        if "probe" in self._loaded:
            return self._loaded["probe"]
        with self._load_lock:
            if "probe" in self._loaded:
                return self._loaded["probe"]
            probe = None
            if os.path.isfile(settings.PROBE_PATH):
                try:
                    with open(settings.PROBE_PATH) as f:
                        raw = json.load(f)
                    groups = sorted(raw["groups"])
                    w = np.asarray([raw["groups"][g]["w"] for g in groups], np.float32)
                    b = np.asarray([raw["groups"][g]["b"] for g in groups], np.float32)
                    if w.ndim != 2 or w.shape[0] != len(groups):
                        raise ValueError("bad weight shape")
                    probe = {"groups": groups, "w": w, "b": b, "version": raw.get("version")}
                except (OSError, ValueError, KeyError, TypeError) as e:
                    self._errors["probe"] = f"invalid probe file: {e}"
                    log.error("probe file %s unusable: %s", settings.PROBE_PATH, e)
            self._loaded["probe"] = probe
            return probe

    def loaded_models(self):
        return [k for k in UNLOADABLE if k in self._loaded]

    def unload(self):
        """Drop every loaded model and give the memory back (idle unload). The caller makes sure
        no request is using them; the next request loads them again. Returns what was dropped."""
        with self._load_lock, self.ocr_lock, self.yunet_lock:
            dropped = [k for k in UNLOADABLE if self._loaded.pop(k, None) is not None]
        gc.collect()
        trim_memory()
        if dropped:
            log.info("unloaded %s after idling", ", ".join(dropped))
        return dropped

    # Embeddings ------------------------------------------------------------------------------
    def siglip_embed(self, images, boxes=None):
        """L2-normalised SigLIP 2 image vectors (pooler_output): one per image, then one per box of the first."""
        x = [to_tensor(im, SIGLIP_SIZE, SIGLIP_RESAMPLE, SIGLIP_MEAN, SIGLIP_STD) for im in images]
        x += [to_tensor(images[0], SIGLIP_SIZE, SIGLIP_RESAMPLE, SIGLIP_MEAN, SIGLIP_STD, b) for b in boxes or []]
        x = np.stack(x)
        out = self.siglip().run(["pooler_output"], {"pixel_values": x})[0]
        return l2(out.astype(np.float32))

    def dino_embed(self, images, boxes=None):
        """L2-normalised DINOv2 class tokens, one per image then one per box of the first.
        last_hidden_state[:, 0, :] equals pooler_output (phase 0, point 6)."""
        x = [to_tensor(im, DINO_SIZE, DINO_RESAMPLE, DINO_MEAN, DINO_STD) for im in images]
        x += [to_tensor(images[0], DINO_SIZE, DINO_RESAMPLE, DINO_MEAN, DINO_STD, b) for b in boxes or []]
        x = np.stack(x)
        out = self.dino().run(["last_hidden_state"], {"pixel_values": x})[0]
        return l2(out[:, 0, :].astype(np.float32))

    # Health ----------------------------------------------------------------------------------
    def status(self):
        kinds = {"siglip": "siglip", "dino": f"dino-{self.dino_variant}", "yunet": "yunet"}
        out = {}
        for name, kind in kinds.items():
            if kind in self._loaded or name in self._loaded:
                out[name] = "loaded"
            elif kind in self._errors or name in self._errors:
                out[name] = "error"
            else:
                out[name] = self.file_state(kind)
        out["ocr"] = "loaded" if "ocr" in self._loaded else ("error" if "ocr" in self._errors else "in_image")
        p_ok = os.path.isfile(os.path.join(settings.PROMPTS_DIR, "prompts.npy"))
        out["prompts"] = "loaded" if "prompts" in self._loaded else ("present" if p_ok else "missing")
        if "probe" in self._loaded:
            out["probe"] = "loaded" if self._loaded["probe"] else ("invalid" if "probe" in self._errors else "absent")
        else:
            out["probe"] = "present" if os.path.isfile(settings.PROBE_PATH) else "absent"
        return out

    def errors(self):
        return dict(self._errors)
