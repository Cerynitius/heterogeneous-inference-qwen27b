# Usage

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
| `ROUTER_HOST` | 0.0.0.0 | router listen address |
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

`reasoning_effort`: `low`, `medium` (default), `high`, `xhigh`, `max`. Default: `router/adapd.py --default-effort`; thinking
budget per tier: `--think-caps`.

Several agents with the same first message: send a distinct `X-Session-Id` header per agent.

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

Set in the environment of `scripts/serve.sh`. Under `scripts/duo.sh`, change them in its `COMMON` / `LAY` arrays.

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
kernels/    fp8attn, fp4_gemv, gdn_defer (C++/CUDA, JIT-built when vLLM imports them), decode_fuse (Triton)
router/     adapd.py — session homing + adaptive P/D router
scripts/    serve.sh (one server), duo.sh (two cards + router)
tools/      drafter W4A16 builders, draft vocabulary tools
assets/     draft vocabulary
```
