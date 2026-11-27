# vision: fast photo checks

> Copied from the Dlala API repository (the `vision/` folder). « Phase N » and « note 40 » refer to the internal
> design plan, summarised in [`../docs/SYSTEM_DESIGN.md`](../docs/SYSTEM_DESIGN.md). Paths such as `config/photo-checks.json`
> or `services/*.ts` name files of the Dlala API, which is not in this repository.

A private Python service, beside `clamd`: no published port, an internal Docker network with no
route to the internet, no secret, no state. Each request carries its image(s); nothing is
stored or logged except one access line (method, route, status, time). Standard-library
`ThreadingHTTPServer`, ONNX Runtime and OpenCV on the CPU. The small vision-language model is a
separate service, [`vlm`](vlm/README.md).

Models (licences and sha256 in [`models.lock.json`](models.lock.json)): SigLIP 2 base 224
(vision tower only), DINOv2 small (or base, `DINO_VARIANT=base`), YuNet 2023mar, PP-OCRv6 small
inside the `rapidocr` wheel. They are not in the image: `python -m app.fetch_models` (run by the deployment's
fetch script) puts them in the `vision_models` volume. Each one loads on its first use. After `VISION_IDLE_UNLOAD_S` (600 s)
without a request they all leave memory (about 0.6 GB back to the server when nobody sells) and the
next request loads them again (a few seconds more for that request).

## Request format

`POST` routes take one of:

- `Content-Type: application/x-dlala-frames`: 4 bytes (big-endian length N of the JSON), N bytes
  of UTF-8 JSON (an object: the options below, plus `"lengths": [bytes of image 1, bytes of image 2]`),
  then the image bytes one after the other. Node: `Buffer.concat([u32be(json.length), json, ...images])`.
- `Content-Type: image/*` or `application/octet-stream`: the body is one image, no options.

Images: JPEG (and MPO), PNG, WebP; Pillow is not allowed to parse any other format (TIFF, BMP,
GIF... are refused unread). At most 4096 x 4096 pixels (the originals of phase 2) and 8192 px a
side; at least 16 px. Body at most 32 MiB.

Boxes, in requests and answers, are `[x0, y0, x1, y1]`: four finite numbers, in pixels of the
**upright** image (EXIF orientation applied: the same frame as the `size` field of the answers),
`x0, y0` included and `x1, y1` excluded (a box covering the whole image is `[0, 0, width, height]`).
Every JSON answer of a `POST` route carries `took_ms` (computing time, without the wait for a slot).

| Route | Images | Options | Answer (JSON unless said) |
|---|---|---|---|
| `POST /fingerprint` | 1 | `regions`: up to 8 boxes (photo rectangles of a screenshot, phase 4), each at least 32 px a side; `tiles`: `true` adds 5 fixed query tiles after them (the 2 x 2 quadrants, then the centre half; phase 4b), none when the image is under 64 px a side | `pdq`: 8 hex hashes (64 characters) in the order of `orientations` (original, rotate90, rotate180, rotate270, flipx, flipy, flipplus1, flipminus1); `quality`; `dino`: base64 of little-endian float16, L2-normalised (`dino_dim` 384 for small, 768 for base); `dino_variant`; `size`; `regions`: `[{box, pdq, quality, dino, tile}]` (`tile`: one of the fixed tiles) |
| `POST /screen` | 1 | `parts`: any of `siglip`, `hotspots`, `ocr`, `faces`, `photo_rects` (default: all but `hotspots`); `tiles`: grid for hotspots, 0 to 3 (default 2: 4 tiles) | `size`: `[width, height]` of the upright image; `siglip`: score per restricted group (`ivory`, `protected_species`, `weapons`, `counterfeit`, `cultural_goods`, `human_remains`, `nazi`); `siglip_ordinary`: best "ordinary object" prompt; `siglip_top`: the 3 best prompts; `siglip_vector`: base64 float16, 768 values (for the probe and § 6); `probe`: logistic-probe probability per group, `null` while `app/data/probe.json` does not exist; `hotspots`: `[{box, scores{group}}]`, best tile first (only when asked); `ocr`: `[{text, score, box, quad}]`; `faces`: `[{box, landmarks[5], score}]` (score >= 0.5); `photo_rects`: `[{box, score, natural}]` (at most 4, see below); `screenshot_signs`: `{phone_aspect, screen_size, status_bar, status_bar_text, flat_fraction, text_fraction, text_lines, photo_rect_fraction, collage_cells}` (`collage_cells`: how many photo cells tile the image, 0 when it is not a collage; `null` without `photo_rects`); `timings_ms` |
| `POST /verify` | 2 | `box_a`, `box_b`: compare only these parts | `inliers` (SIFT + `cv2.findHomography(..., cv2.RANSAC)`; 0 when the homography crushes or folds the first picture or scales its area outside 1/400 to 400, else counted once per distinct point on each side; the mirrored second image is tried below 60 inliers or when the homography found is a reflection, so the API's `verify.minInliers` must stay under 60), `matches`, `keypoints`, `mirrored` (the copy is a mirror image: the mirrored try won, or the homography kept is a reflection), `homography` (`null` when rejected), `method`, `evidence_jpeg` (base64: both photos side by side with the inlier lines) |
| `POST /redact` | 1 | `boxes`: 1 to 32 boxes | `image/jpeg` (quality 90), the boxes blurred after growing by 15 % around their centre; header `X-Redacted-Boxes`: the blurred boxes |
| `GET /health` | | | `{status, api_version: "vision-v1", dino_variant, models{...}, errors, uptime_s, rss_mb, peak_rss_mb}`; 503 when a model file is missing or failed |

Scores are raw. Every threshold that turns a score into a decision is the API's
(`config/photo-checks.json`, phase 1; not in this repository). SigLIP 2 score of a prompt:
`sigmoid(cos(image, prompt) * exp(logit_scale) + logit_bias)` with the checkpoint's constants
(4.7245, -16.7717); a group's score is the maximum over its prompts. Prompts:
[`tools/build-prompts.py`](tools/build-prompts.py), vectors in `app/data/prompts.npy`. Production uses 73 prompts
(7 groups and "ordinary" negatives); **they are private**. This copy ships a sample of 10 prompts (one per group,
three negatives; `"sample": true` in `app/data/prompts.json`) that shows the format: the service runs on it and the
tests pass, but its scores are not the production screen's.

Photo rectangles (`photo_rects`, phase 4b, `app/screen.py`): on a copy of at most 800 px, three sources of
candidates, then one filter (between 3 % and 92 % of the image, sides at most 4 : 1, at least 128 colours on
4 bits a channel) and the 4 best by area x colours. (1) Textured blobs (local spread of the grey level >= 3
over 7 x 7 px); a blob that is not a full rectangle counts when 3 of its box's sides are lined with texture
(a photo is cut by straight lines; an object on a white page is not). The OCR boxes, when `ocr` was asked
too, are removed from the texture first. (2) Edge quadrilaterals (Canny). (3) Recursive cuts on a copy of at
most 400 px (the XY-cut of page-layout analysis): a part is cut along a uniform band across it (a page margin,
a collage gutter; the page's own colour first) or, for the first two levels, along a straight colour step
across 70 % of it (two photos side by side); the leaves are trimmed of their uniform margins, and a trimmed
leaf whose trimmed sides do not look like a photo's edges falls back to the box between its cut lines (a
photo whose own white or black border was trimmed). Measured in the calibration report of 2026-11-25 ([`../docs/ML_CV.md`](../docs/ML_CV.md));
A long thin image (130 : 1 and more) skips the cuts. The original is never copied at full size (Pillow's
integer box reduction first): about 50 ms of CPU at 1,600 px and 1.4 times that at 4,096 px on the work
laptop; `tests/test_photo_rects.py` bounds both as ratios (against a fixed reference workload, and 4,096 px
against 1,600 px), so the check does not depend on the machine.

Hotspots cost one SigLIP pass per tile. Ask for them (`{"parts": ["hotspots"]}`) only for the
photos that go to the small model; they are the close-ups to crop from the 4096 px original.

`probe.json` (phase 1, optional): `{"version", "groups": {"<group>": {"w": [768 floats], "b": float}}}`;
probability = `sigmoid(w . siglip_vector + b)`. An unreadable file gives `probe: null` and an `errors.probe` in `/health`.

Errors: `{"error": {"code", "message"}}`. 400 `empty_body`, `empty_image`, `bad_frames`, `bad_meta`,
`image_too_small`, `short_body`; 404 `not_found`; 411 `length_required`; 413 `body_too_large`,
`image_too_large`; 415 `not_an_image`, `bad_content_type`; 503 `busy` (one request is computed at
a time; at most `MAX_WAITING` (4) wait, each up to `QUEUE_WAIT_S` (60 s); a fifth is refused at
once, before its body is read), `model_unavailable` (never a result without the model); 500
`internal` (logged; the process keeps running).

## Settings (environment)

`MODELS_DIR` (/models), `DINO_VARIANT` (small), `ORT_THREADS` (1), `MAX_INFLIGHT` (1), `MAX_WAITING` (4),
`QUEUE_WAIT_S` (60), `VISION_IDLE_UNLOAD_S` (600; 0 keeps the models loaded), `MAX_BODY_BYTES` (32 MiB),
`OCR_DET_SIDE` (736), `OCR_MAX_SIDE` (1280), `VERIFY_SHA_ON_LOAD` (1), `PROBE_PATH`.

**OCR sizes: phase 1 must check the OCR's accuracy at 736 / 1280** (accented French, ID cards,
plates, site names in screenshots) and raise them if text is missed: at 960 / 1600 the OCR took
about 120 MB more. ONNX Runtime's telemetry is switched off (`onnxruntime.disable_telemetry_events()`).

## int8

Plan rule: keep int8 only if its cosine to fp32 is >= 0.99 on 20 images. Measured on the 20 test
images (tests/fixtures.json), against the fp32 ONNX:

| Model | published `_int8` (S8 weights) | published `_uint8` (U8U8) | own `quantize_dynamic`, MatMul only, per channel |
|---|---|---|---|
| SigLIP 2 vision | min 0.53, mean 0.70 | min 0.52, mean 0.70 | min 0.53, mean 0.71 |
| DINOv2-S | min 0.91, mean 0.98 | min 0.87, mean 0.97 | min 0.90, mean 0.98 |
| DINOv2-B | min 0.09, mean 0.18 | min 0.07, mean 0.18 | not tried |

All fail, so the service runs fp32. U8U8 failing too rules out the AVX2 saturation that ONNX
Runtime documents for U8S8 (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
these vision transformers have activation outliers that per-tensor dynamic quantisation cannot hold.
The ARM kernels differ, but the outliers do not: no reason to expect better there.

## Measurements (work laptop, amd64, i5-8265U, `--cpus 0.75`, one thread, shared with other jobs: load 3 to 4)

ONNX fp32 against PyTorch (transformers 5.24, torch 2.15): cosine 1.000000 (min) on the 20 images for
SigLIP 2, DINOv2-S and DINOv2-B.

Process memory (VmRSS; peak = VmHWM), loading lazily in this order: idle 62 MB; after DINOv2-S
192 MB (load peak 225); after YuNet 273 MB; after SigLIP 2 613 MB (load peak 807; loading holds a
second copy of the weights for a moment, given back by `malloc_trim`); after RapidOCR 740 MB.
Steady with everything loaded: 680 MB. Peak over the whole run, 4096 px photos included: 944 MB
(cgroup `memory.peak` 875 MiB). Hence `mem_limit: 1200m` in the compose file ([`../deploy/compose.example.yml`](../deploy/compose.example.yml)).

After the review of 24/11 (OCR 736 / 1280, idle unload): the OCR step peaks at 786 MB instead of
830; the whole run's peak is 937 MB (cgroup 902 MiB; reached by `/redact` or `/fingerprint` on a
4096 px original, not by the OCR); after the idle unload the process holds 122 MB (cgroup 86 MiB).
`/screen` on a 1600 px photo: 3.7 s (OCR 2.0 s instead of 3.2 s). The table below is from before.

| Call | p50 | p95 |
|---|---|---|
| `/fingerprint` 1600 px | 0.63 s | 0.77 s |
| `/fingerprint` 4096 px + 1 region | 1.68 s | 1.70 s |
| `/screen` 1600 px, default parts (SigLIP 1.2 s, OCR 3.2 s, faces 0.3 s, rectangles 0.06 s) | 4.7 s | 4.8 s |
| `/screen` 4096 px, default parts | 4.2 s | 4.2 s |
| `/screen` hotspots only (5 SigLIP passes) | 4.9 s | 5.1 s |
| `/verify` 1600 px pair | 1.7 s | 1.8 s |
| `/redact` 4096 px, 1 box | 1.1 s | 1.2 s |

OCR detection: RapidOCR's default (shorter side at least 736 px) ran detection on 2000 x 1600 px:
11.9 s and +700 MB on a 10 MP photo. With the longest side at most 1280: 3.7 s, +390 MB;
at 960: 2.8 s, +300 MB. Default since the review of 24/11: detection at 736 on a copy of at most 1280 px.

PDQ (rule: distance <= 10 at quality >= 80): the 8 bridge images of quality 100 are within 2 to 8
bits of the C++ reference output; the other images of quality 100 match ThreatExchange's own
test values exactly; the 8 dihedral hashes are within 10 bits of `pdq-photo-hasher --pdqdih`;
hashing a copy of at most 1024 px moves a hash by at most 8 bits.

## Tests

    MODELS_DIR=<folder with vision/...> service/tests/run.sh

Builds the image, runs `tests/` inside it (46 tests, about 2 minutes here): PDQ compliance, ONNX
against the PyTorch reference, `/redact` hides YuNet faces, every route on bad input, the
answers of each route, the photo rectangle finder (pages, collages, its time). Test images download once (sha256-checked, `tests/fixtures.json`).
`tests/measure.py` prints the memory and p50/p95 of a running service (its docstring says how to
run it on the server). Fill a local models folder with
`python3 -m app.fetch_models --lock models.lock.json --dest <folder> --groups vision vision-dino-base` (from `service/`).

Work-computer tools (need torch and transformers, internet; nothing of them enters the image):
`tools/build-prompts.py` (the prompt vectors), `tools/make-reference.py` (the PyTorch reference).
They read only the needed tensors of the safetensors files, into memory.

## Still to measure on the ARM server

`pdqhash` compiling on arm64 (the image build in CI does it), the image size, memory and p50/p95
(`tests/measure.py`), whether `/screen` fits the phase 1 target of 5 s per photo with
`cpus: 0.75`, and the bid simulations' latency while photos are checked (plan phase 3).
