"""Compute the SigLIP 2 text vectors of the screening prompts, once, on a work computer.

    python tools/build-prompts.py [--prompts <file.json>]   (from service/; needs torch, transformers, internet)

Writes app/data/prompts.npy (float32, one L2-normalised row per prompt) and prompts.json
(the prompts in the same order, their group, the model revision, logit_scale and logit_bias).
The vision service never loads the 1.1 GB text tower: it only compares image vectors with these.
Text rules (phase 0, point 3): lower case, padding to max_length 64, truncation.
Only the text tower's tensors and the embedding rows of the tokens used are downloaded, into memory.
"""
import datetime
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_safetensors import RemoteSafetensors  # noqa: E402

REPO = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors"
MAX_LENGTH = 64
# Phase 0, point 3: read from this checkpoint's header; the script stops if they differ.
EXPECTED_LOGIT_SCALE = 4.7245
EXPECTED_LOGIT_BIAS = -16.7717
CONSTANT_TOLERANCE = 1e-3
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data")

# SAMPLE prompts: one plain description per restricted group, and a few ordinary objects as negatives.
# The production prompt set is private (it would show what the screen looks for and which look-alikes it
# separates). To build your own, pass --prompts <file.json> holding {"<group>": ["<text>", ...], ...};
# every group of app/screen.py RESTRICTED_GROUPS needs at least one prompt.
SAMPLE_PROMPTS = {
    "ivory": ["a photo of a carved elephant ivory figurine"],
    "protected_species": ["a photo of a tiger skin rug"],
    "weapons": ["a photo of a handgun"],
    "counterfeit": ["a photo of a luxury designer handbag with a monogram logo"],
    "cultural_goods": ["a photo of an ancient egyptian artifact"],
    "human_remains": ["a photo of a human skeleton"],
    "nazi": ["a photo of a swastika"],
    # Negatives: what ordinary lots look like.
    "ordinary": ["a photo of a wooden chair", "a photo of a porcelain vase", "a photo of a framed painting"],
}


def load_prompts(argv):
    """(prompts, is_sample): the --prompts file when given, else the sample above."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", help='JSON file {"<group>": ["<text>", ...]}; default: the built-in sample')
    a = ap.parse_args(argv)
    if not a.prompts:
        return SAMPLE_PROMPTS, True
    with open(a.prompts) as f:
        prompts = json.load(f)
    if not isinstance(prompts, dict) or not all(isinstance(v, list) and v and all(isinstance(t, str) for t in v) for v in prompts.values()):
        raise SystemExit("--prompts must hold an object of non-empty lists of strings")
    return prompts, False


def main(argv=None):
    PROMPTS, is_sample = load_prompts(argv)
    import torch
    from transformers import AutoTokenizer, SiglipConfig, SiglipTextModel

    texts = [(g, t.lower()) for g, ts in PROMPTS.items() for t in ts]
    with tempfile.TemporaryDirectory() as tmp:
        tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, cache_dir=tmp)
        cfg = SiglipConfig.from_pretrained(REPO, revision=REVISION, cache_dir=tmp)
    ids = tok([t for _, t in texts], padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="np")["input_ids"]
    used = sorted(set(int(i) for i in ids.ravel()))

    st = RemoteSafetensors(URL)
    scale = float(st.tensor("logit_scale")[0])
    bias = float(st.tensor("logit_bias")[0])
    assert abs(scale - EXPECTED_LOGIT_SCALE) < CONSTANT_TOLERANCE and abs(bias - EXPECTED_LOGIT_BIAS) < CONSTANT_TOLERANCE, (scale, bias)
    emb_name = "text_model.embeddings.token_embedding.weight"
    weights = st.tensors([n for n in st.names("text_model.") if n != emb_name])
    rows = st.rows(emb_name, used)
    print(f"downloaded the text tower without its embedding table, and {len(used)} of its rows", flush=True)

    with torch.device("meta"):
        model = SiglipTextModel(cfg.text_config)
    # transformers 4.x nests the tower under .text_model; 5.x does not (checkpoint keys keep the prefix).
    inner = getattr(model, "text_model", model)
    prefix = "" if inner is not model else "text_model."
    # A compact embedding table: the used rows only, the token ids remapped to it.
    inner.embeddings.token_embedding = torch.nn.Embedding(len(used), rows.shape[1], device="meta")
    state = {k[len(prefix):]: torch.from_numpy(v) for k, v in weights.items()}
    state[emb_name[len(prefix):]] = torch.from_numpy(rows)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # position_ids is a buffer the model builds itself.
    assert not unexpected and all("position_ids" in m for m in missing), (missing, unexpected)
    # Buffers not stored in the checkpoint (position_ids) are still on the meta device: rebuild them.
    if inner.embeddings.position_ids.is_meta:
        inner.embeddings.register_buffer("position_ids", torch.arange(MAX_LENGTH).unsqueeze(0), persistent=False)
    model.eval()
    remap = {t: i for i, t in enumerate(used)}
    local = torch.tensor([[remap[int(i)] for i in row] for row in ids])
    with torch.no_grad():
        vec = model(input_ids=local).pooler_output.float().numpy()
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "prompts.npy"), vec.astype(np.float32))
    meta = {
        **({"about": "SAMPLE prompt set, not the production one (tools/build-prompts.py without --prompts). Score of an image for one prompt: sigmoid(cos(image, prompt) * exp(logit_scale) + logit_bias); a group's score is the maximum over its prompts. 'ordinary' prompts are negatives.", "sample": True} if is_sample else
           {"about": "SigLIP 2 text vectors of the screening prompts (tools/build-prompts.py). Score of an image for one prompt: sigmoid(cos(image, prompt) * exp(logit_scale) + logit_bias); a group's score is the maximum over its prompts. 'ordinary' prompts are negatives."}),
        "model": REPO, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
        "text": {"lowercase": True, "padding": "max_length", "max_length": MAX_LENGTH, "truncation": True},
        "logit_scale": scale, "logit_bias": bias,
        "built": datetime.date.today().isoformat(),
        "torch": torch.__version__,
        "prompts": [{"group": g, "text": t} for g, t in texts],
    }
    with open(os.path.join(OUT_DIR, "prompts.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(texts)} prompts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
