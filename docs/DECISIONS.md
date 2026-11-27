# Decisions

Short records of the choices that shaped the system. Sources: [S] [`service/README.md`](../service/README.md);
[V] [`service/vlm/README.md`](../service/vlm/README.md); [P] the internal design note; [R2] the calibration
report of 2026-11-25 (see [`ML_CV.md`](ML_CV.md)).

**Contents**

1. [No paid AI](#1-no-paid-ai)
2. [A local cascade](#2-a-local-cascade)
3. [fp32, not int8](#3-fp32-not-int8)
4. [A standard-library HTTP server](#4-a-standard-library-http-server)
5. [Models in a volume, not in the image](#5-models-in-a-volume-not-in-the-image)
6. [One request at a time](#6-one-request-at-a-time)
7. [Shadow before on](#7-shadow-before-on)
8. [A legal guard before any automatic refusal](#8-a-legal-guard-before-any-automatic-refusal)
9. [The uploader owns the photo](#9-the-uploader-owns-the-photo)

## 1. No paid AI

- **Context.** A large hosted vision model reads photos well, but costs money on every photo (estimated at
  0.20 to 0.70 $ for a sale of 10 photos [P § 2]) and sends members' photos to a provider. The owner refused the
  cost.
- **Decision.** Only open models whose licences allow commercial use, running on Dlala's own server.
- **Consequences.** 0 € and no photo leaves the server. A 2-billion-parameter model recognises materials less
  well than a large one; its accuracy per group must be measured, and is not yet.

## 2. A local cascade

- **Context.** On one CPU the small model takes 22 to 112 s per answer on the laptop, depending on the image
  size [V]: it cannot look at every photo.
- **Decision.** Cheap stages (PDQ, DINOv2, SigLIP 2, OCR, YuNet) see every photo; the small model sees only the
  photos they flag, with one closed question per call.
- **Consequences.** The fast filters' thresholds decide how much work reaches the model; they must be calibrated
  on the « objets » set before the screen flags anything on its own.

## 3. fp32, not int8

- **Context.** int8 weights would cut the vision models' memory about fourfold. The rule: keep int8 only if its
  cosine to fp32 is at least 0.99 on 20 images [S].
- **Measured** [S]:

  | Model | published `_int8` (S8 weights) | published `_uint8` (U8U8) | own `quantize_dynamic`, MatMul only, per channel |
  |---|---|---|---|
  | SigLIP 2 vision | min 0.53, mean 0.70 | min 0.52, mean 0.70 | min 0.53, mean 0.71 |
  | DINOv2-S | min 0.91, mean 0.98 | min 0.87, mean 0.97 | min 0.90, mean 0.98 |
  | DINOv2-B | min 0.09, mean 0.18 | min 0.07, mean 0.18 | not tried |

- **Decision.** fp32. U8U8 failing too rules out the AVX2 saturation that ONNX Runtime documents for U8S8: these
  vision transformers have activation outliers that per-tensor dynamic quantisation cannot hold [S].
- **Consequences.** Memory is spent elsewhere instead: DINOv2-S rather than B, models loaded on first use and
  unloaded when idle.

## 4. A standard-library HTTP server

- **Context.** The service computes one request at a time and answers a single private client.
- **Decision.** Python's `ThreadingHTTPServer`, no web framework [S].
- **Consequences.** Less to install and less memory: the idle process holds 62 MB [S]. Framing, limits, error
  codes and back-pressure are written out in [`server.py`](../service/app/server.py) and tested route by route.

## 5. Models in a volume, not in the image

- **Context.** The model files weigh about 2.0 GB ([`models.lock.json`](../service/models.lock.json): SigLIP 2
  371.8 MB, DINOv2-S 88.5 MB, YuNet 0.2 MB, Qwen3-VL-2B 1,107.4 MB and its encoder 445.1 MB). The image is rebuilt
  at every release; the server's disk is 45 GB and was once full [P].
- **Decision.** Images carry code only. A fetch step downloads each model once into a volume, checks its size and
  sha256, and the services mount it read-only; each model file is hashed again when it loads [S].
- **Consequences.** Releases stay small and offline; a new model is a change of the lock file. The fetch is the
  only moment a container reaches the internet.

## 6. One request at a time

- **Context.** `vision` has 0.75 of a CPU. A second request in parallel adds memory, not speed [S].
- **Decision.** One request computed at a time, at most 4 waiting for at most 60 s each; beyond that, 503 `busy`
  before the body is read. `vlm` runs with `-np 1`.
- **Consequences.** Memory stays under its measured peak; the API's client queues its own calls and retries a
  busy answer later.

## 7. Shadow before on

- **Context.** Thresholds measured on evaluation sets may behave differently on real listings.
- **Decision.** `PHOTO_CHECKS=shadow` computes and stores every new decision while the old review still decides;
  `on` acts. The plan switches to `on` only after a shadow week whose figures meet fixed criteria (not built
  yet: the operator sets it).
- **Consequences.** Rules that lift a guardrail (near identity: a copy without the inlier count) run in shadow
  first and only record what they would have matched [R2 § 4].

## 8. A legal guard before any automatic refusal

- **Context.** GDPR art. 22 allows a decision based solely on automated processing only on one of its grounds,
  here the contract, with safeguards; the terms of use in force said the machine never refuses [P].
- **Decision.** The code runs `on` as `shadow` until the terms in force provide for automatic decisions. Every
  restriction carries a statement of reasons (DSA art. 17); a contest goes to a person.
- **Consequences.** Turning the checks on is a legal step (new terms published by the owner) before it is a
  technical one.

## 9. The uploader owns the photo

- **Context.** A copy detector can be turned against honest sellers: file a request with someone's photo
  address, or upload a scraped photo first, and the real owner's listing becomes « a copy ».
- **Decision.** Context rules, computed from the data, decide whether a confirmed copy counts:
  - the seller's own photos are not even compared: relisting is never a reason;
  - when the other record's address was uploaded by the seller, it never counts;
  - only a use another account cannot stage counts: another account's sale that went live, or its request once
    approved. A refused, pending, withdrawn or lapsed request never counts;
  - a photo that is in no such sale or request is never a copy, else whoever uploads a scraped photo first would
    get its real owner flagged;
  - when the other photo was itself proved to copy one of the seller's listed photos, the seller's is the
    original;
  - a seller who won and paid for the lot that shows the photo is reselling it: ignored;
  - a suspended sale of another account showing it is a serious reason; another member's live sale or approved
    request, a refusal reason.
- **Consequences.** Only ids are kept, never the other member's identity, and the seller never sees who.
