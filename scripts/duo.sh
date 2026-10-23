#!/bin/bash
# Two cards behind router/adapd.py (OpenAI API on ROUTER_PORT, default 8090). Runs until Ctrl-C / SIGTERM.
#   scripts/duo.sh [apd|home]
#   apd   (default) sessions homed per card + adaptive P/D for a session's first (cold) prefill: the prefill may run
#         on the other card and its KV is pushed over NIXL (UCX, cuda_ipc); DFlash2-W4A16 drafter (k = 7)
#   home  two independent replicas, sessions homed per card, MTP drafter (k = 4), no KV transfer
# Environment:
#   MODEL          target checkpoint dir (required; home: an -mtpw4a16 build, tools/build_mtp_w4a16.py)
#   DRAFTER_MODEL  apd: DFlash2 W4A16 drafter dir (required; tools/build_dflash2_w4a16.py)
#   GPU0, GPU1     card A (PRO 6000 class) and card B (PRO 5000 class): index or UUID (0, 1)
#   KV0, KV1       KV cache GiB per card (66, 42)          SEQS0, SEQS1   max sequences per card (10, 10)
#   CPUS0, CPUS1   taskset core lists per server (unset = no pinning); CPUSR for the router
#   PORT0, PORT1   server ports (8000, 8001)               ROUTER_PORT    (8090)
#   CF0, CF1       scheduler cost constants "us_per_token ns_per_token_ctx"
#                  (apd: "87.2 0.870", "103.0 1.123"; home: "77.8 1.425", "95.4 1.828")
#   ROUTER_ARGS    extra router/adapd.py arguments          LOGDIR        (./logs)
#   any other variable (QWEN38_*, B12X_DIR, VLLM, ...) is passed to scripts/serve.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LAYOUT=${1:-apd}
: "${MODEL:?MODEL=<checkpoint dir> is required}"
GPU0=${GPU0:-0}; GPU1=${GPU1:-1}
KV0=${KV0:-66}; KV1=${KV1:-42}; SEQS0=${SEQS0:-10}; SEQS1=${SEQS1:-10}
PORT0=${PORT0:-8000}; PORT1=${PORT1:-8001}; ROUTER_PORT=${ROUTER_PORT:-8090}
LOGDIR=${LOGDIR:-$PWD/logs}; mkdir -p "$LOGDIR"
VLLM=${VLLM:-vllm}
PY=$(dirname "$(command -v "$VLLM")")/python
[[ -x $PY ]] || PY=python3
TAG=$(date +%m%d_%H%M%S)
pin() { [[ -n ${1:-} ]] && echo "taskset -c $1"; }

COMMON=(NVFP4=b12x FP8=cutlass QWEN38_SCHED=1 B12XPF=1 QWEN38_GDN_DEFER=1 QWEN38_ACTQ=1 BLOCK=1664 MNBT=8192
        QWEN38_SCHED_TURGENT_MS=0)
case $LAYOUT in
  apd)
    : "${DRAFTER_MODEL:?DRAFTER_MODEL=<DFlash2 W4A16 drafter dir> is required for apd}"
    kvx="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\"]}}"
    NIXL=(UCX_TLS=all VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_SSM_CONV_STATE_LAYOUT=DS "KVX=$kvx")
    read -ra CF0 <<< "${CF0:-87.2 0.870}"; read -ra CF1 <<< "${CF1:-103.0 1.123}"
    LAY=(DRAFTER=dflash2 SPEC=7 QWEN38_KV_GROUP_SIZE=16 QWEN38_FP8ATTN=1 QWEN38_FP8DEC=1 QWEN38_SCHED_ALIGN=auto
         QWEN38_SCHED_TADD_MS=1000 QWEN38_SCHED_TSOLO_MS=500
         QWEN38_SCHED_PF_CURVE=1 QWEN38_SCHED_PF_S0=1.0 QWEN38_SCHED_PF_SINF=0.4 QWEN38_SCHED_PF_K_S=15 QWEN38_SCHED_PF_GAMMA=1
         QWEN38_SCHED_DEC_FLOOR_TPS=20 QWEN38_SCHED_DEC_WINDOW_S=10 QWEN38_SCHED_DEC_FLOOR_MIN_BL=0 QWEN38_SCHED_DEC_TAU=2.99
         "${NIXL[@]}")
    E0=(VLLM_NIXL_SIDE_CHANNEL_PORT=5600 QWEN38_SCHED_ELIN_NS=6092 QWEN38_SCHED_EATT_PS=585.3)
    E1=(VLLM_NIXL_SIDE_CHANNEL_PORT=5601 QWEN38_SCHED_ELIN_NS=7844 QWEN38_SCHED_EATT_PS=785.8)
    VIS0=$GPU0,$GPU1; VIS1=$GPU1,$GPU0           # each server sees both cards (own first) for cuda_ipc
    TOKPG=26000; RMODE=(--home --home-split --tadd-share 0.9) ;;
  home)
    read -ra CF0 <<< "${CF0:-77.8 1.425}"; read -ra CF1 <<< "${CF1:-95.4 1.828}"
    LAY=(SPEC=4 QWEN38_ARGMAX=1 "DRAFT_VOCAB=${DRAFT_VOCAB-$HERE/assets/draft_vocab_98304.pt}" QWEN38_SCHED_TADD_MS=1500 QWEN38_SCHED_TSOLO_MS=1500)
    E0=(QWEN38_SCHED_ELIN_NS=6092 QWEN38_SCHED_EATT_PS=585.3); E1=(QWEN38_SCHED_ELIN_NS=7844 QWEN38_SCHED_EATT_PS=785.8)
    VIS0=$GPU0; VIS1=$GPU1
    TOKPG=27600; RMODE=(--home --no-split) ;;
  *) echo "unknown layout $LAYOUT (apd | home)"; exit 1 ;;
esac

for p in $PORT0 $PORT1 $ROUTER_PORT; do
  ss -ltn | grep -qE ":$p\b" && { echo "port $p already in use"; exit 1; }
done
for g in $GPU0 $GPU1; do
  u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ $u -lt 1000 ]] || { echo "GPU $g busy: $u MiB used"; exit 1; }
done

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  for i in $(seq 1 60); do
    alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    (( alive )) || break; sleep 1
  done
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start() {  # start NAME VISIBLE PORT CPUS KVGIB SEQS CLIN CATT ENV...
  local name=$1 vis=$2 port=$3 cpus=$4 kv=$5 seqs=$6 clin=$7 catt=$8; shift 8
  local log=$LOGDIR/serve_${TAG}_$name.log
  setsid $(pin "$cpus") env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$vis" PORT="$port" KVGIB="$kv" MAXSEQS="$seqs" \
    QWEN38_SCHED_CLIN_US="$clin" QWEN38_SCHED_CATT_NS="$catt" "${COMMON[@]}" "${LAY[@]}" "$@" \
    "$HERE/scripts/serve.sh" > "$log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "== $name: card $vis port $port, log $log"
  until grep -qE "Application startup complete|Engine core initialization failed" "$log"; do
    kill -0 "${PIDS[-1]}" 2>/dev/null || break; sleep 5
  done
  grep -q "Application startup complete" "$log" || { echo "$name failed to start:"; tail -20 "$log"; exit 1; }
}
# one after the other: weight loading and compilation peak in host memory
start a "$VIS0" "$PORT0" "${CPUS0:-}" "$KV0" "$SEQS0" "${CF0[0]}" "${CF0[1]}" "${E0[@]}"
start b "$VIS1" "$PORT1" "${CPUS1:-}" "$KV1" "$SEQS1" "${CF1[0]}" "${CF1[1]}" "${E1[@]}"

RLOG=$LOGDIR/router_$TAG.log
$(pin "${CPUSR:-}") "$PY" "$HERE/router/adapd.py" --host "${ROUTER_HOST:-0.0.0.0}" --port "$ROUTER_PORT" --log "$RLOG" "${RMODE[@]}" \
  --backend "a=http://127.0.0.1:$PORT0,${CF0[0]},${CF0[1]},$SEQS0,$("$PY" -c "print(int($KV0*$TOKPG))")" \
  --backend "b=http://127.0.0.1:$PORT1,${CF1[0]},${CF1[1]},$SEQS1,$("$PY" -c "print(int($KV1*$TOKPG))")" \
  ${ROUTER_ARGS:-} >> "$RLOG" 2>&1 &
PIDS+=($!)
for i in $(seq 1 60); do curl -s --noproxy '*' -m 2 "http://127.0.0.1:$ROUTER_PORT/v1/models" | grep -q qwen3.8 && break; sleep 1; done
curl -s --noproxy '*' -m 2 "http://127.0.0.1:$ROUTER_PORT/v1/models" | grep -q qwen3.8 || { echo "router did not come up:"; tail -20 "$RLOG"; exit 1; }
echo "== ready: http://127.0.0.1:$ROUTER_PORT/v1  (layout $LAYOUT)"
wait -n "${PIDS[@]}"
echo "a server or the router exited; stopping"
