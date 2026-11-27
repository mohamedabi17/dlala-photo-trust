"""PyTorch reference vectors for the ONNX test (plan phase 3: cosine >= 0.999 on 20 images).

    python tools/make-reference.py --cache <fixture cache dir>   (needs torch, transformers, internet)

For the 20 photo and face fixtures of tests/fixtures.json, computes with transformers (the
reference implementation) the L2-normalised image vectors of SigLIP 2 (vision tower,
pooler_output) and of DINOv2 small and base (pooler_output), and writes them to
tests/data/reference-<model>.npy. The images go through app.images.decode first, so both sides
start from the same RGB pixels; the preprocessing is the library's own (no centre crop for DINOv2,
as the service does, plan phase 3). Weights are read into memory only.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VISION = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, VISION)
from remote_safetensors import RemoteSafetensors  # noqa: E402
from app.images import decode  # noqa: E402
from tests.fixtures import fixture_path, load_fixtures  # noqa: E402

MODELS = {
    "siglip": {"repo": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", "prefix": "vision_model."},
    "dino-small": {"repo": "facebook/dinov2-small", "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056", "prefix": ""},
    "dino-base": {"repo": "facebook/dinov2-base", "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "prefix": ""},
}
OUT = os.path.join(VISION, "tests", "data")
RENAMES = [("attention.attention.query", "attention.q_proj"), ("attention.attention.key", "attention.k_proj"),
           ("attention.attention.value", "attention.v_proj"), ("attention.output.dense", "attention.o_proj")]


def build(name, spec, tmp):
    import torch
    from transformers import AutoConfig, AutoImageProcessor, Dinov2Model, SiglipVisionModel

    cfg = AutoConfig.from_pretrained(spec["repo"], revision=spec["revision"], cache_dir=tmp)
    if name == "siglip":
        cls, cfg = SiglipVisionModel, cfg.vision_config
    else:
        cls = Dinov2Model
    with torch.device("meta"):
        model = cls(cfg)
    st = RemoteSafetensors(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/model.safetensors")
    names = st.names(spec["prefix"])
    # transformers 5.x drops the "vision_model." level that the checkpoint keys still carry.
    strip = spec["prefix"] if not hasattr(model, "vision_model") else ""
    state = {k[len(strip):]: torch.from_numpy(v) for k, v in st.tensors(names).items()}
    # transformers 5.x renamed DINOv2's attention weights (its conversion_mapping.py does the same).
    own = set(model.state_dict())
    for k in list(state):
        if k not in own:
            new = k
            for a, b in RENAMES:
                new = new.replace(a, b)
            if new in own:
                state[new] = state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    assert not unexpected, unexpected
    for buf_name, buf in list(model.named_buffers()):
        if buf.is_meta:  # buffers rebuilt by the model, not stored (position_ids)
            mod_name, _, leaf = buf_name.rpartition(".")
            mod = model.get_submodule(mod_name)
            mod.register_buffer(leaf, torch.arange(buf.shape[-1]).expand(buf.shape).clone(), persistent=False)
    assert all("position_ids" in m for m in missing), missing
    model.eval()
    try:
        proc = AutoImageProcessor.from_pretrained(spec["repo"], revision=spec["revision"], cache_dir=tmp, use_fast=False)
    except (TypeError, ValueError):
        proc = AutoImageProcessor.from_pretrained(spec["repo"], revision=spec["revision"], cache_dir=tmp)
    kwargs = {} if name == "siglip" else {"do_center_crop": False, "size": {"height": 224, "width": 224}}
    return model, proc, kwargs, type(proc).__name__


def main():
    import tempfile

    import torch
    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="fixture cache directory (downloaded if empty)")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    fixtures = [f for f in load_fixtures() if f["kind"] in ("photo", "face")]
    images = [decode(open(fixture_path(f, a.cache), "rb").read()) for f in fixtures]
    meta = {"about": __doc__.strip().splitlines()[0], "built": datetime.date.today().isoformat(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "fixtures": [f["id"] for f in fixtures], "models": {}}
    meta_path = os.path.join(OUT, "reference.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta["models"] = json.load(fh).get("models", {})
    for name in a.models:
        spec = MODELS[name]
        with tempfile.TemporaryDirectory() as tmp:
            model, proc, kwargs, proc_name = build(name, spec, tmp)
        with torch.no_grad():
            pixel_values = proc(images=images, return_tensors="pt", **kwargs)["pixel_values"]
            out = model(pixel_values=pixel_values).pooler_output.float().numpy()
        vec = out / np.linalg.norm(out, axis=1, keepdims=True)
        np.save(os.path.join(OUT, f"reference-{name}.npy"), vec.astype(np.float32))
        meta["models"][name] = {"repo": spec["repo"], "revision": spec["revision"], "processor": proc_name, "processor_kwargs": kwargs, "dim": int(vec.shape[1])}
        print(f"{name}: {vec.shape} written", flush=True)
        del model
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
