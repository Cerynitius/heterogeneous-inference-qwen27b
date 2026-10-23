#!/bin/bash
# One vLLM server for Qwen3.8-27B NVFP4 on the visible card(s). All settings come from the environment:
#   MODEL         checkpoint dir (required)                       PORT      (8000)
#   KVGIB         KV cache size in GiB (31.5)                     MAXSEQS   max concurrent sequences (6)
#   MNBT          max batched tokens per step (32768)             BLOCK     attention block size (vLLM default)
#   SPEC          draft tokens, 0 = no speculative decoding (0)
#   DRAFTER       mtp | dflash2 (mtp)                             DRAFTER_MODEL  DFlash2 drafter dir (dflash2 only)
#   DRAFT_VOCAB   pruned draft vocabulary .pt (MTP only)          DRAFT_SAMPLE   probabilistic = sampled drafts (dflash2)
#   PREFIX        1 = prefix caching (1)                          PMU       --prefix-match-unit
#   NVFP4         b12x | vllm_cutlass (FlashInfer default)        FP8       cutlass (FlashInfer default)
#   B12XPF        1 = b12x prefill attention (B12X_DIR)        KVX       --kv-transfer-config JSON
#   OFFLOAD_GIB   CPU tier for the prefix cache (GiB)             TEXTONLY  1 = no vision encoder
#   PP            pipeline parallel size (1)                      EAGER     1 = --enforce-eager
#   GMU           --gpu-memory-utilization (0.92)                 AUTOTUNE  1 = FlashInfer autotune
#   COMPILE_CACHE 1 = keep the torch.compile cache                PROF      torch profiler output dir
#   THINK_END     text forced at the thinking budget ("" = off)
#   VLLM          vllm executable (vllm on PATH)                  QWEN38_*  passed through to the patched vLLM
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL:?MODEL=<checkpoint dir> is required}"
PORT="${PORT:-8000}"; SPEC="${SPEC:-0}"; PREFIX="${PREFIX:-1}"
GMU="${GMU:-0.92}"; KVGIB="${KVGIB:-31.5}"
VLLM="${VLLM:-vllm}"
PY=$(dirname "$(command -v "$VLLM")")/python
[[ -x $PY ]] || PY=python3
export QWEN27B_KERNELS="${QWEN27B_KERNELS:-$HERE/kernels}"
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
[[ ${COMPILE_CACHE:-0} != 1 ]] && export VLLM_DISABLE_COMPILE_CACHE=1
[[ ${AUTOTUNE:-0} != 1 ]] && NOTUNE=(--no-enable-flashinfer-autotune) || NOTUNE=()

EXTRA=()
DIS=()
[[ ${NVFP4:-} == b12x ]] && DIS+=(FlashInferCuteDslNvFp4LinearKernel FlashInferCutlassNvFp4LinearKernel)
[[ ${NVFP4:-} == vllm_cutlass ]] && DIS+=(FlashInferCuteDslNvFp4LinearKernel FlashInferCutlassNvFp4LinearKernel FlashInferB12xNvFp4LinearKernel)
[[ ${FP8:-} == cutlass ]] && DIS+=(FlashInferFP8ScaledMMLinearKernel)
[[ ${#DIS[@]} -gt 0 ]] && export VLLM_DISABLED_KERNELS=$(IFS=,; echo "${DIS[*]}")

# scheduler cost model (QWEN38_SCHED=1): per-card defaults unless set
card=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=name --format=csv,noheader 2>/dev/null)
if [[ ${QWEN38_SCHED:-0} == 1 && -z ${QWEN38_SCHED_CLIN_US:-} ]]; then
  if [[ $card == *"PRO 5000"* ]]; then export QWEN38_SCHED_CLIN_US=103.0 QWEN38_SCHED_CATT_NS=1.123
  else export QWEN38_SCHED_CLIN_US=87.2 QWEN38_SCHED_CATT_NS=0.870; fi
fi
if [[ ${QWEN38_SCHED_ALIGN:-} == auto ]]; then
  [[ $card == *"PRO 5000"* ]] && export QWEN38_SCHED_ALIGN=576 || export QWEN38_SCHED_ALIGN=832
fi

LAR=""
if [[ -n ${DRAFT_VOCAB:-} ]]; then export QWEN38_DRAFT_VOCAB=$DRAFT_VOCAB; LAR=',"use_local_argmax_reduction":true'; fi
if [[ ${DRAFTER:-mtp} == dflash2 ]]; then
  : "${DRAFTER_MODEL:?DRAFTER_MODEL=<DFlash2 drafter dir> is required with DRAFTER=dflash2}"
  SPEC=${SPEC/#0/7}; unset QWEN38_DRAFT_VOCAB
  EXTRA+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER_MODEL\",\"num_speculative_tokens\":$SPEC,\"disable_eagle_block_drop\":true${DRAFT_SAMPLE:+,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}}")
else
  [[ $SPEC != 0 ]] && EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC$LAR}")
fi
[[ $PREFIX == 1 ]] && EXTRA+=(--enable-prefix-caching)
[[ -n ${PMU:-} ]] && EXTRA+=(--prefix-match-unit "$PMU")
[[ -n ${PROF:-} ]] && EXTRA+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"active_iterations\":20}")
if [[ -n ${OFFLOAD_GIB:-} ]]; then
  OFF="{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$("$PY" -c "print(int($OFFLOAD_GIB*2**30))")}}"
  if [[ -n ${KVX:-} ]]; then
    KVX="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[$KVX,$OFF]}}"
  else
    KVX=$OFF
  fi
fi
[[ -n ${KVX:-} ]] && EXTRA+=(--kv-transfer-config "$KVX")
[[ ${EAGER:-0} == 1 ]] && EXTRA+=(--enforce-eager)
[[ ${TEXTONLY:-0} == 1 ]] && EXTRA+=(--language-model-only)
if [[ ${B12XPF:-0} == 1 ]]; then
  export QWEN38_B12X_PREFILL=1 PYTHONPATH="${B12X_DIR:-$HERE/third_party/b12x}${PYTHONPATH:+:$PYTHONPATH}"
fi
[[ -n ${BLOCK:-} ]] && EXTRA+=(--block-size "$BLOCK")
[[ ${PP:-1} != 1 ]] && EXTRA+=(--pipeline-parallel-size "$PP")
THINK_END=${THINK_END-$'\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n'}
[[ -n $THINK_END ]] && EXTRA+=(--reasoning-config "$("$PY" -c 'import json,sys; print(json.dumps({"reasoning_end_str": sys.argv[1]}))' "$THINK_END")")

exec "$VLLM" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host "${HOST:-0.0.0.0}" --port "$PORT" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --seed 0 \
  --gpu-memory-utilization "$GMU" --kv-cache-memory-bytes "$("$PY" -c "print(int($KVGIB*2**30))")" \
  --max-num-seqs "${MAXSEQS:-6}" \
  --max-num-batched-tokens "${MNBT:-32768}" \
  --enable-chunked-prefill "${NOTUNE[@]}" \
  "${EXTRA[@]}"
