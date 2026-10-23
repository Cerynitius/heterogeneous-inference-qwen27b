# heterogeneous-inference-qwen27b

Qwen3.8-27B (NVFP4) served on two different Blackwell cards — reference setup: RTX PRO 6000 (96 GB) + RTX PRO 5000 (72 GB) — with vLLM 0.30.0, custom sm_120a kernels, a latency-aware prefill scheduler and an adaptive prefill/decode router.

## Performance

Measured on RTX PRO 6000 (power limit 400 W) + RTX PRO 5000, `scripts/duo.sh apd` defaults unless noted. Wait = request sent to first token; e2e = request sent to last token; decode tok/s = output tokens after the first / their time.

**10 concurrent requests × 256K context** (10 sessions sent at once, ~248K-token prompts, 1024 output tokens each)

| | cold: nothing cached (2.48M prompt tokens) | warm: history cached, +4K new tokens each (253K context) |
|---|---|---|
| wait mean / worst | 203 / 339 s | 7.8 / 11.2 s |
| e2e mean / worst | 219 / 348 s | 24.4 / 27.0 s |
| e2e tok/s per request mean / worst | 5.7 / 2.9 | 42.1 / 38.0 |
| decode tok/s per request mean / worst | 70 / 43 | 63 / 49 |
| aggregate prefill (both cards) | 7,300 tok/s | |
| aggregate output, average / peak 5 s | 29 / 127 tok/s | 376 / 629 tok/s |
| total time | 348 s | 27 s |

**Single request, idle card** (one server per card, nothing cached; two requests per length)

| prompt tokens | wait PRO 6000 (mean / worst) | prefill PRO 6000 (mean / worst) | wait PRO 5000 (mean / worst) | prefill PRO 5000 (mean / worst) |
|---|---|---|---|---|
| 4K | 0.3 / 0.3 s | 12,370 / 12,205 tok/s | 0.4 / 0.4 s | 10,043 / 9,689 tok/s |
| 16K | 1.3 / 1.4 s | 12,150 / 11,314 tok/s | 1.6 / 1.7 s | 10,100 / 9,385 tok/s |
| 32K | 2.8 / 2.9 s | 11,263 / 10,898 tok/s | 3.5 / 3.6 s | 9,272 / 8,944 tok/s |
| 64K | 6.5 / 6.6 s | 9,777 / 9,753 tok/s | 8.1 / 8.1 s | 7,936 / 7,910 tok/s |
| 128K | 16.3 / 16.3 s | 7,867 / 7,843 tok/s | 20.2 / 20.2 s | 6,337 / 6,326 tok/s |
| 192K | 29.2 / 29.6 s | 6,573 / 6,489 tok/s | 36.6 / 36.9 s | 5,253 / 5,208 tok/s |
| 250K | 46.9 / 48.2 s | 5,330 / 5,186 tok/s | 58.6 / 60.3 s | 4,273 / 4,147 tok/s |

| single-stream decode (DFlash2 drafter, ~1K output tokens, 12 prompts) | mean | worst | best |
|---|---|---|---|
| PRO 6000 | 236 tok/s | 133 tok/s | 314 tok/s |
| PRO 5000 | 181 tok/s | 108 tok/s | 270 tok/s |
| wait (short prompt) | 0.05 s | 0.07 s | |

**Agent load: 10 concurrent coding-agent sessions, 15 min** (contexts growing from 15K–223K up to 259K; 3 runs, range)

| | value |
|---|---|
| requests per run | 416–465 |
| wait mean | 3.1–3.6 s |
| wait p90 | 4.7–5.1 s |
| wait worst (cold 160K-token session start under load) | 104–122 s |
| wait mean, follow-up turns | 2.2–2.6 s |
| session start (cold prefill) under load, 16K / 64K / 128K / 224K | 1.5–1.6 / 12.8–15.9 / 55.0–56.9 / 69.4–75.4 s |
| turn e2e mean / worst | 15.7–17.4 s / 108–163 s |
| decode tok/s per stream mean / p10 / worst | 59–66 / 32–35 / 16–17 |
| aggregate output, run average | 345–378 tok/s |
| aggregate output, peak 60 s | 509–523 tok/s |

## Requirements

- Linux, Python 3.12, NVIDIA driver with CUDA 13 support
- Two sm_120 GPUs (RTX PRO 6000 / 5000 class)
- CUDA toolkit 13.x (`nvcc`) for the JIT-built kernels: set `CUDA_HOME` and put `$CUDA_HOME/bin` on `PATH`

## Install

```bash
git clone git@github.com:Cerynitius/heterogeneous-inference-qwen27b.git
cd heterogeneous-inference-qwen27b
./install.sh                     # VENV=.venv PYTHON=python3.12 by default
source .venv/bin/activate
```

`install.sh` installs `vllm==0.30.0`, `nixl[cu13]`, `aiohttp`, copies `vllm/` over the installed vLLM package and puts `b12x==1.3.0` into `third_party/b12x`.

## Weights

```bash
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir weights/Qwen3.8-27B-NVFP4
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir weights/Qwen3.8-27B-DFlash2

python tools/build_mtp_w4a16.py    weights/Qwen3.8-27B-NVFP4     # -> weights/Qwen3.8-27B-NVFP4-mtpw4a16
python tools/build_dflash2_w4a16.py weights/Qwen3.8-27B-DFlash2   # -> weights/Qwen3.8-27B-DFlash2-w4a16
```

The build tools need one GPU.

## Run

Both cards behind the router (OpenAI API on port 8090):

```bash
export MODEL=$PWD/weights/Qwen3.8-27B-NVFP4-mtpw4a16
export DRAFTER_MODEL=$PWD/weights/Qwen3.8-27B-DFlash2-w4a16
GPU0=0 GPU1=1 scripts/duo.sh apd     # sessions homed per card + adaptive P/D over NIXL, DFlash2 drafter
GPU0=0 GPU1=1 scripts/duo.sh home    # two replicas, sessions homed per card, MTP drafter
```

`GPU0` is the larger card, `GPU1` the smaller one (index or UUID). Ctrl-C stops the servers and the router. Logs go to `./logs`.

| variable | default | |
|---|---|---|
| `KV0`, `KV1` | 66, 42 | KV cache GiB per card |
| `SEQS0`, `SEQS1` | 10, 10 | max sequences per card |
| `PORT0`, `PORT1`, `ROUTER_PORT` | 8000, 8001, 8090 | |
| `CPUS0`, `CPUS1`, `CPUSR` | unset | `taskset` core lists for the servers / router |
| `CF0`, `CF1` | per layout | scheduler cost constants `"us_per_token ns_per_token_ctx"` |
| `ROUTER_ARGS` | | extra `router/adapd.py` arguments |
| `LOGDIR` | `./logs` | |

One server on one card:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=$PWD/weights/Qwen3.8-27B-NVFP4-mtpw4a16 \
  SPEC=4 DRAFT_VOCAB=$PWD/assets/draft_vocab_98304.pt NVFP4=b12x B12XPF=1 QWEN38_SCHED=1 \
  KVGIB=60 scripts/serve.sh
```

All `scripts/serve.sh` settings are listed at the top of the script.

## Request

```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "max_tokens": 4096,
  "reasoning_effort": "medium"
}'
```

`reasoning_effort`: `low`, `medium`, `high`, `xhigh`, `max` (thinking budget per tier: `router/adapd.py --think-caps`).

Images and video: OpenAI content parts (`image_url`, `video_url`; data URLs or http(s) links). The router shrinks images above 4,194,304 pixels before forwarding (`ROUTER_ARGS="--img-max-pixels N"`, `0` = off).

```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "What is in this image?"}]}]
}'
```

## Feature flags

Set in the environment of `scripts/serve.sh` (`scripts/duo.sh` sets them per layout).

| flag | |
|---|---|
| `QWEN38_SCHED=1` | latency-aware prefill scheduler (`QWEN38_SCHED_*` tunables: `vllm/v1/core/sched/qwen38_sched.py`) |
| `QWEN38_SCHED_ELIN_NS`, `QWEN38_SCHED_EATT_PS` | vision-encoder cost per image patch (linear, quadratic) for the scheduler |
| `QWEN38_FP8ATTN=1` | FP8 prefill attention kernel (needs `B12XPF=1`) |
| `QWEN38_FP8DEC=1` | FP8 spec-verify decode attention |
| `QWEN38_GDN_DEFER=1` | deferred GDN state writes for speculative decoding |
| `QWEN38_ACTQ=1` | fused SiLU-and-mul + NVFP4 activation quant |
| `QWEN38_ARGMAX=1` | draft argmax kernel (MTP) |
| `QWEN38_FP4_GEMV=1` | small-M NVFP4 GEMV for the MLP (needs `NVFP4=b12x`) |
| `QWEN38_KV_GROUP_SIZE=n` | KV cache group size |
| `QWEN38_DRAFT_VOCAB=file` | pruned draft vocabulary (`DRAFT_VOCAB` in `serve.sh`) |

## Draft vocabulary

`assets/draft_vocab_98304.pt` is the default for the MTP drafter. To rebuild:

```bash
python tools/corpus_freq.py <corpus dir> <tokenizer.json> <code dir> vocab/corpus_counts.npz
python tools/vocab_freq.py --base http://127.0.0.1:8000 --tag w0      # -> vocab/gen_w0.jsonl
python tools/vocab_select.py vocab/gen_w0.jsonl --tokenizer <tokenizer.json> --write 98304
```

## Layout

```
vllm/       modified vLLM 0.30.0 files (installed over the vLLM package)
kernels/    fp8attn, fp4_gemv, gdn_defer, decode_fuse (JIT-built on first use)
router/     adapd.py — session homing + adaptive P/D router
scripts/    serve.sh (one server), duo.sh (two cards + router)
tools/      drafter W4A16 builders, draft vocabulary tools
assets/     draft vocabulary
```

## License

Apache-2.0. Files under `vllm/` are modified from vLLM 0.30.0 (Apache-2.0); see `NOTICE`.
