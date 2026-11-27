# vlm: the small vision-language model

llama.cpp's own server image, pinned by digest, running **Qwen3-VL-2B-Instruct** Q4_K_M with its
Q8_0 `mmproj` (Apache-2.0; files, sizes and sha256 in [`../models.lock.json`](../models.lock.json),
fetched into the `vision_models` volume by `app.fetch_models`). No code of ours runs in it.
It sits on the internal `vision` network with `vision` and the two APIs: no published port, no
internet, no secret, read-only filesystem, user `nobody`.

## Command line (production; [`../../deploy/compose.example.yml`](../../deploy/compose.example.yml))

Image `ghcr.io/ggml-org/llama.cpp:server-b12679@sha256:6fab63852916c294cc8a2f1515079c4ca40248193eea8868ae742134335b989b`
(multi-platform index; arm64 manifest `sha256:5a5dd769...`). Its entrypoint is `/app/llama-server`
and it sets `LLAMA_ARG_HOST=0.0.0.0`, `LLAMA_ARG_PORT=8080`, so `VLM_URL=http://vlm:8080`.

    /app/llama-server \
      -m /models/vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
      --mmproj /models/vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
      -np 1 -c 2048 -t 1 -tb 1 -cram 0 --image-max-tokens 512 \
      --no-repack --no-ui --sleep-idle-seconds 300

| Flag | Why | Source (server README at `b12679`) |
|---|---|---|
| `-m`, `--mmproj` | the model and its image encoder | l. 180 |
| `-np 1` | one request at a time | l. 178 |
| `-c 2048` | short context: an overview and two close-ups of at most 512 tokens each, the question (~150 tokens) and the answer (<= 120) fit; the KV cache costs 112 KiB a token (28 layers x 8 KV heads x 128 x 2 x f16), 224 MiB here instead of the model's 256k default | l. 50 |
| `-t 1 -tb 1` | one thread for generation and for reading the image and prompt | l. 38-39 |
| `-cram 0` | no prompt cache in RAM (8 192 MiB by default) | l. 169 |
| `--image-max-tokens 512` | bounds each image (Qwen3-VL: one token per 32 x 32 px), so a large photo cannot overflow the context or the memory | l. 186 |
| `--no-repack` | keeps the Q4_K weights in the mapped file instead of a private repacked copy: 0.81 GB private memory instead of 1.70 GB, answers about 40 % slower (measured below). Memory is this server's hard limit; the small model only sees flagged photos | l. 69 |
| `--no-ui` | no chat web page | l. 213 |
| `--sleep-idle-seconds 300` | after 5 idle minutes the model and its KV cache leave RAM; the next request reloads them. `GET /health` and `GET /props` do not wake it (README "Sleeping on Idle", l. 2604-2617), so the healthcheck is safe | l. 252 |
| (`--load-mode`) | left at `auto`: mmap on CPU (l. 74); there is no `--mmap` flag any more | l. 74 |

Not used: `--lazy-mode` (only for per-layer embedding tables: Gemma 4).

Container limits (the compose file): `cpus: 1.0` with `cpu_shares: 128` (an eighth of
the API's weight), `mem_limit: 2048m`. The measured peak RSS is 1.90 GB, of which 1.07 GB are the
mapped model pages: under a lower limit the kernel would drop them and read them again from disk
for every token. **On the ARM server, check the major page faults while it answers**
(`grep pgmajfault /sys/fs/cgroup/memory.stat` in the container, before and after 20 answers):
they must stay near zero.

## Calling it (plan phase 5)

`POST /v1/chat/completions`, the image first, then one closed question; the reply is forced by
`response_format` (OpenAI form, accepted by llama.cpp, README l. 1466):

    {"messages": [
       {"role": "system", "content": "<instruction>"},
       {"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
         {"type": "text", "text": "<one closed question>. Reply with JSON only, exactly these keys: {...}"}]}],
     "response_format": {"type": "json_schema", "json_schema": {"name": "photo_vlm_v1", "strict": true, "schema": {...}}},
     "temperature": 0, "max_tokens": 120}

Schema rules, checked against the converter the server uses (`common/json-schema.cpp` at
`b12679`, l. 149-296: `required`, `additionalProperties`, `enum`, `anyOf`, `minItems`/`maxItems`,
`minLength`/`maxLength`, integer `minimum`/`maximum` are all read): box coordinates are integers
0-1000 relative to the image; no `anyOf` beside `properties` in the same object; the format is
also written in the question, since the model never sees the schema. The full object schema and a
sample question are in [`test_schema.py`](test_schema.py). The production instruction and questions are private.

## Test

[`test_schema.py`](test_schema.py) sends the 20 test photos (overview of at most 640 px) with
the object schema and checks every reply parses and fits it. Run it in a container of the
vision image sharing the vlm container's network (its docstring has the command).

## Measurements (work laptop, amd64, i5-8265U, `--cpus 1.0`, shared with other jobs: load 3 to 4)

The model files were on tmpfs here (the laptop's disk was nearly full): load times from the
server's disk will be longer. RSS = VmRSS of llama-server; "private" = RssAnon; the rest is the
mapped GGUF (clean pages the kernel can drop and read again).

| | default (repack) | `--no-repack` (production) |
|---|---|---|
| Start to `/health` 200 | 16.8 s | 10.4 to 20.6 s |
| RSS loaded, idle | 2.48 GB (private 1.40) | 1.83 GB (private 0.74) |
| RSS after answers (peak) | 2.78 GB (private 1.70) | 1.90 GB (private 0.81) |
| RSS asleep (`--sleep-idle-seconds`) | 0.16 GB | 0.08 GB |
| Answer, 256 px product photo (~220 prompt tokens) | 22 to 36 s (12 photos) | 38 to 42 s (3 photos) |
| Answer, 640 px photo (380 to 560 prompt tokens) | 63 to 112 s (8 photos) | not run |
| Wake from sleep + answer | 46 s (reload about 19 s) | 53 s |

Of an answer, reading the image and prompt takes most of the time (one thread): 16 to 23 s for
~220 tokens, 51 to 101 s for 380 to 560 tokens; generating ~40 tokens takes 6 to 12 s. The plan's
target (at most 90 s per flagged photo, overview and up to two close-ups) therefore needs small
images: phase 1 should try overviews of about 384 to 448 px (150 to 200 image tokens) and close-ups
of the same size; at 640 px a single image already takes 60 to 110 s here.

Schema: 20 of 20 replies parsed and fitted it (default build), 3 of 3 with `--no-repack`, and the
reply after waking from sleep too. `GET /health` answers 200 while asleep without waking it.
Content (not this phase's measure): the ivory netsuke were all "yes" with a box; two ordinary
products (a glass lamp, a cream object) were also "yes": the false positives phase 1 measures.

## Still to measure on the ARM server

Load and reload time from the volume's disk, time per answer at the overview size phase 1 picks
(p50/p95 over 20 photos), RSS loaded and asleep, and the bid simulations' latency during 20
answers (plan phase 3). Repacking matters more there: llama.cpp repacks Q4_K for ARM's
dot-product instructions.
