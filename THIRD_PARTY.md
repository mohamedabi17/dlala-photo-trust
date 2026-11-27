# Third-party components

None of these files is in this repository: models and images are downloaded by sha256 from their sources
([`service/models.lock.json`](service/models.lock.json), [`service/tests/fixtures.json`](service/tests/fixtures.json),
the manifests in [`eval/`](eval/)). Each keeps its own licence. Licence names below are quoted from the
lock file, the sources' model cards or package metadata, as checked on 2026-11-24 unless stated.

## Models

| Component | Used for | Licence | Source |
|---|---|---|---|
| SigLIP 2 base patch16 224, vision tower (ONNX) | zero-shot screen | Apache-2.0 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) (model card: `license: apache-2.0`); ONNX export [onnx-community/siglip2-base-patch16-224-ONNX](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX), which declares no licence and derives from it |
| DINOv2 small (ONNX); DINOv2 base, optional | copy-detection embedding | Apache-2.0 | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base); ONNX exports [onnx-community/dinov2-small](https://huggingface.co/onnx-community/dinov2-small), [onnx-community/dinov2-base](https://huggingface.co/onnx-community/dinov2-base) (no licence declared, derived) |
| YuNet `face_detection_yunet_2023mar.onnx` | face detection | MIT (« MIT License, Copyright (c) 2020 Shiqi Yu ») | [opencv/opencv_zoo, face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| Qwen3-VL-2B-Instruct, `Q4_K_M` GGUF and `mmproj` `Q8_0` | small vision-language model | Apache-2.0 | [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (model card: `license: apache-2.0`) |
| PP-OCRv6 small (detection, recognition) and the PP-OCR mobile orientation classifier, inside the `rapidocr` wheel | text reading | Apache-2.0 | [PaddlePaddle/PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec), PP-OCRv6_small_det; PaddleOCR is Apache-2.0 |

Not used, by licence: SSCD and the ISC21 winner (weights trained on DISC21, CC BY-NC 4.0).

## Software

| Component | Version | Licence | Source |
|---|---|---|---|
| llama.cpp `llama-server` image | `server-b12679`, pinned by digest | MIT | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/b12679/docs/docker.md) |
| PDQ (perceptual hash) | ThreatExchange `pdq/` | BSD | [facebook/ThreatExchange](https://github.com/facebook/ThreatExchange/tree/main/pdq) |
| `pdqhash` (Python bindings of PDQ) | 0.2.8 | MIT | [PyPI](https://pypi.org/project/pdqhash/) |
| ONNX Runtime | 1.32.0 | MIT License | [PyPI](https://pypi.org/project/onnxruntime/) |
| OpenCV (`opencv-python-headless`; SIFT's patent expired in March 2020) | 4.14.0.94 | Apache 2.0 | [PyPI](https://pypi.org/project/opencv-python-headless/) |
| RapidOCR | 3.10.0 | Apache-2.0 | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) |
| Pillow | 13.0.0 | MIT-CMU | package metadata |
| NumPy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | package metadata |
| Shapely | 2.2.0 | BSD-3-Clause | package metadata |
| pyclipper | 1.4.0 | MIT | package metadata |
| omegaconf | 2.4.0 | BSD License | package metadata |
| PyYAML | 6.0.4 | MIT | package metadata |
| requests, urllib3, idna, charset-normalizer, certifi | see `service/requirements.lock` | Apache-2.0, MIT, BSD-3-Clause, MIT, MPL-2.0 | package metadata |
| protobuf, flatbuffers, packaging, colorlog, tqdm, six | see `service/requirements.lock` | 3-Clause BSD, Apache 2.0, Apache-2.0 OR BSD-2-Clause, MIT, MPL-2.0 AND MIT, MIT | package metadata |
| Python base image | `python:3.12-slim`, pinned by digest | PSF and the Debian packages' licences | Docker Hub |
| sharp (evaluation scripts only) | 0.35.5 | Apache-2.0 | npm |

Package metadata was read from the built `vision` image (`importlib.metadata`) on 2026-11-27.

## Datasets and images

| Source | Used for | Licence |
|---|---|---|
| [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | phase 1 fingerprint set (10,000 photos), 1,000 ordinary objects, 12 test fixtures | CC BY 4.0, credit Amazon.com |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), validation split | phase 4b fingerprint set (4,000 photos) | images listed by the dataset as CC BY 2.0, each with its Flickr author in `eval/openimages-manifest.json`; annotations CC BY 4.0 (Google LLC) |
| [Wikimedia Commons](https://commons.wikimedia.org/) | the « objets » set (restricted objects, look-alikes, personal data and traps, demo-library objects), test fixtures | per image, as Commons states it: public domain, CC0, CC BY or CC BY-SA; licence and author recorded per file in `eval/vision-manifest.json` and `service/tests/fixtures.json` |
| ThreatExchange `pdq/data` | PDQ compliance tests | BSD (ThreatExchange repository) |
| NASA portraits on Commons | face tests | public domain (NASA) |

Derived images (transformed copies, injection images) are made locally by the scripts, never committed, and
keep their parent's licence.
