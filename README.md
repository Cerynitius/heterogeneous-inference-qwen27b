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

## What's inside

- **One endpoint over two different cards.** Agent sessions live on one card each, placed by free KV and card speed; a
  session's cold prefill may run on whichever card finishes it first, and its KV is pushed over NIXL to the home card.
- **Latency-aware prefill scheduler.** Long prefills are cut into time-capped chunks so running decodes keep streaming;
  the shortest remaining work goes first, image prefills ahead of text.
- **Custom sm_120a kernels.** FP8 prefill attention, small-M NVFP4 GEMV, deferred GDN state stores for speculative
  decoding, fused SiLU-and-mul + NVFP4 quant, draft argmax.
- **Speculative decoding.** DFlash2 drafter (NVFP4 W4A16) in the `apd` layout, MTP with a pruned draft vocabulary in `home`.
- **Claude-style serving on an OpenAI API.** Thinking tiers with hard budgets (default `medium`), chain of thought never
  returned or kept in the context, compaction at the context end, large images shrunk before prefill.

## Usage

Install, weights, launch, requests and flags: [docs/USAGE.md](docs/USAGE.md).

## License

Apache-2.0. Files under `vllm/` are modified from vLLM 0.30.0 (Apache-2.0); see `NOTICE`.
