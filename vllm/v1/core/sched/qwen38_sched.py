# hetero-qwen27b: latency-aware prefill scheduling for the Qwen3.8-27B server (QWEN38_SCHED=1).
#
# Stock vLLM fills each step's 32768-token budget in `running` order, so one long prefill chunk (up to ~14 s at 256K on
# the PRO 6000) holds every decode behind it and keeps waiting requests (even fully prefix-cached ones) out until it is
# done. Rules here, after NInfer's "decode every round, bounded prefill per round" (but prefill still rides in the same
# forward as the decodes):
#   1. decodes first: `running` is reordered every step, decodes in admission order, then prefills by least remaining
#      prefill time (SRPT);
#   2. a step that carries decodes gets at most T_ADD seconds of prefill work, priced by cost() (position-dependent
#      attention term); a step without decodes gets at most T_SOLO seconds, so a request that arrives during a long
#      prefill waits at most about one T_SOLO step to be admitted (stock: up to a whole 32K chunk, ~14 s at 256K);
#   3. while a request waits with a free seat, running prefills leave RESERVE tokens and RESERVE_FRAC of T_ADD to it,
#      so it is admitted in the same step and SRPT takes over from the next one;
#   4. progress: the first running prefill and every admitted request get at least MIN_CHUNK tokens.
# Chunk sizes below the Mamba block (1600 tokens) are passed to the align-mode split as its chunk limit, which then
# advances sub-block and re-aligns at the next block boundary (stock behaviour for blocks wider than a chunk).
# Off (default) = stock scheduling.
import collections
import math
import os
import time

ON = os.environ.get("QWEN38_SCHED", "0") == "1"
T_ADD = float(os.environ.get("QWEN38_SCHED_TADD_MS", "40")) / 1e3
T_SOLO = float(os.environ.get("QWEN38_SCHED_TSOLO_MS", "500")) / 1e3 or math.inf  # 0 = full token budget
# cost(start, n) = n * (C_LIN + C_ATT * (start + n/2)) seconds; defaults = PRO 6000 refit
# (32K 5.0 s, 128K 35.1 s, 256K 111.0 s; PRO 5000: 127 us / 3.06 ns from 5.8 / 43.1 / 138.5 s).
C_LIN = float(os.environ.get("QWEN38_SCHED_CLIN_US", "77.8")) / 1e6
C_ATT = float(os.environ.get("QWEN38_SCHED_CATT_NS", "1.425")) / 1e9
MIN_CHUNK = int(os.environ.get("QWEN38_SCHED_MIN_CHUNK", "64"))
# ENC (image input): the vision encoder runs inside the step whose chunk first reaches an image's placeholder
# tokens, and its time is not in cost() (a 16.7 MP image held every decode ~3 s on the PRO 6000). Per image of p patches
# (= 4 x its embedding tokens, spatial merge 2x2): enc = p * (E_LIN + E_ATT * p); the vision attention is per frame, so for
# video the quadratic term uses min(p, ENC_FRAME). Charged with the chunk, deducted from the step's time before the chunk
# is sized, and an image that no longer fits ends the chunk just before it (it is encoded in the next step). 0 = off.
E_LIN = float(os.environ.get("QWEN38_SCHED_ELIN_NS", "0")) / 1e9
E_ATT = float(os.environ.get("QWEN38_SCHED_EATT_PS", "0")) / 1e12
ENC_FRAME = int(os.environ.get("QWEN38_SCHED_ENC_FRAME", "16384"))
LOOKAHEAD = 0   # set by the scheduler: vLLM encodes images up to num_prefill_lookahead tokens past the chunk end
# RESERVE: tokens of the step budget kept for a waiting request. 0 (default) = a quarter of the budget: the
# fixed 8192 dated from max_num_batched_tokens 32768 and with MNBT 8192 left running prefills 0 tokens (and half the
# step's time unused) on every step that admitted a request.
RESERVE = int(os.environ.get("QWEN38_SCHED_RESERVE", "0"))
RESERVE_FRAC = float(os.environ.get("QWEN38_SCHED_RESERVE_FRAC", "0.5"))
# T_URGENT: while something waits on a step boundary -- a request waiting for admission with a free seat, or a
# P/D handover (WAITING_FOR_REMOTE_KVS) whose completion is only noticed between steps -- the step's prefill work is
# capped to T_URGENT seconds instead of T_ADD / T_SOLO, so the event is handled within ~T_URGENT; otherwise steps keep
# the long caps (per-step fixed cost ~25-30 ms on the PRO 6000 makes uniformly short steps cost ~10 % throughput).
T_URGENT = float(os.environ.get("QWEN38_SCHED_TURGENT_MS", "0")) / 1e3   # 0 = off
# ALIGN: a prefill chunk that the time cap cut short is rounded down to a multiple of ALIGN tokens. b12x paged
# extend cost per query token is a 1024-row sawtooth (jumps right after k*1024 rows, lowest just below; 32K/128K/230K
# context alike, kernels/attn_cublas/b12x_nsweep.py) and the b12x FP4 GEMM changes tactic above M=1024, so the
# ~1100-token chunks a 300 ms cap produced ran ~20 % slower per token. 832 = half a 1664-token Mamba block: no
# remainder chunks at block boundaries. 0 = off.
ALIGN = int(os.environ.get("QWEN38_SCHED_ALIGN", "0"))
# DEC_FLOOR: with decodes
# running, a step carries no prefill while the step rate of the last DEC_WINDOW seconds, times DEC_TAU tokens per step
# (spec-decode acceptance), is below DEC_FLOOR tok/s per decoding request. Prefill chunks keep their size (and ALIGN
# efficiency); short decode-only steps are inserted between them. 0 = off.
DEC_FLOOR = float(os.environ.get("QWEN38_SCHED_DEC_FLOOR_TPS", "0"))
DEC_TAU = float(os.environ.get("QWEN38_SCHED_DEC_TAU", "3.0"))
DEC_WINDOW = float(os.environ.get("QWEN38_SCHED_DEC_WINDOW_S", "2.0"))
# DEC_FLOOR_MIN_BL: the floor only acts while the prefill backlog is at least this many cost() seconds, i.e. in
# a real cold burst; a single cold session inside agent load (~55 s of backlog for 223K) must not be slowed (pfC run:
# agent turns -15 %). 0 = always.
DEC_FLOOR_MIN_BL = float(os.environ.get("QWEN38_SCHED_DEC_FLOOR_MIN_BL", "0"))
_step_starts: collections.deque = collections.deque()
# PF_CURVE: with decodes running, the share of
# recent step time spent in steps that carry prefill is capped by a saturating curve of the prefill backlog x (seconds
# of prefill work left in the running set, by cost()):  f(x) = S0 + (SINF - S0) * x^G / (x^G + K^G).  Shortest-work-first
# logic: a small backlog (a turn's few K new tokens) is cheap to finish, so it may take nearly all of the step time
# (S0 ~ 1); a large cold backlog would hold every decode for minutes, so its share falls towards SINF. A step over the
# cap carries no prefill. Values from sched/sim/train_curve.py. 0 = off.
PF_CURVE = os.environ.get("QWEN38_SCHED_PF_CURVE", "0") == "1"
PF_S0 = float(os.environ.get("QWEN38_SCHED_PF_S0", "1.0"))
PF_SINF = float(os.environ.get("QWEN38_SCHED_PF_SINF", "0.7"))
PF_K = float(os.environ.get("QWEN38_SCHED_PF_K_S", "30"))
PF_G = float(os.environ.get("QWEN38_SCHED_PF_GAMMA", "2"))
PF_WINDOW = float(os.environ.get("QWEN38_SCHED_PF_WINDOW_S", "2.0"))   # share window (DEC_WINDOW is the floor's)
_steps: collections.deque = collections.deque()   # [start, duration (filled at the next step), carried prefill]
_last_plan = None
# STEP LOG (curve modelling): QWEN38_SCHED_LOG=<path prefix> appends one JSON line per step to <prefix>.<pid>.jsonl:
# t (step start), dt (start-to-start; with async scheduling ~ the step's duration), dec (decoding requests), dctx (sum of
# their contexts), pf ([start, n] prefill chunks), hold (decode-only step forced by DEC_FLOOR / PF_CURVE), bl (backlog s).
_LOG = os.environ.get("QWEN38_SCHED_LOG", "")
_log_f = None
_log_prev = None


def _log_step(rec):
    global _log_f, _log_prev
    if not _LOG:
        return
    if _log_f is None:
        import json as _json
        _log_f = (open(f"{_LOG}.{os.getpid()}.jsonl", "a", buffering=1), _json)   # line-buffered: survives a hard stop
    f, js = _log_f
    if _log_prev is not None:
        _log_prev["dt"] = round(rec["t"] - _log_prev["t"], 5)
        f.write(js.dumps(_log_prev) + "\n")
    _log_prev = rec


def prefill_share_cap(backlog_s: float) -> float:
    x = max(backlog_s, 0.0) ** PF_G
    return PF_S0 + (PF_SINF - PF_S0) * x / (x + PF_K ** PF_G)


def over_prefill_share(decoding: bool, backlog_s: float) -> bool:
    """Record this step; True when recent prefill time share exceeds the curve (only matters with decodes running)."""
    now = time.monotonic()
    if _steps:
        last = _steps[-1]
        last[1] = now - last[0]
        last[2] = bool(_last_plan is not None and _last_plan.carried_prefill)
    _steps.append([now, 0.0, False])
    while len(_steps) > 1 and now - _steps[0][0] > PF_WINDOW:
        _steps.popleft()
    if not (PF_CURVE and decoding) or len(_steps) < 3:
        return False
    done = list(_steps)[:-1]
    total = sum(d for _, d, _ in done)
    pf = sum(d for _, d, p in done if p)
    return total > 0 and pf / total > prefill_share_cap(backlog_s)


def decode_starved(decoding: bool, backlog_s: float = 0.0) -> bool:
    """Record this step; True when the recent step rate cannot give decodes DEC_FLOOR tok/s."""
    now = time.monotonic()
    _step_starts.append(now)
    while _step_starts and now - _step_starts[0] > DEC_WINDOW:
        _step_starts.popleft()
    if not (DEC_FLOOR > 0 and decoding) or len(_step_starts) < 3 or backlog_s < DEC_FLOOR_MIN_BL:
        return False
    span = max(now - _step_starts[0], 0.25)
    return DEC_TAU * (len(_step_starts) - 1) / span < DEC_FLOOR


def cost(start: int, n: int) -> float:
    """Seconds to prefill positions [start, start + n)."""
    return n * (C_LIN + C_ATT * (start + n / 2))


def tokens_for(start: int, t: float) -> int:
    """Largest n with cost(start, n) <= t."""
    if t <= 0:
        return 0
    if math.isinf(t):
        return 1 << 30
    b = C_LIN + C_ATT * start
    if C_ATT <= 0:
        return int(t / b)
    return int((math.sqrt(b * b + 2 * C_ATT * t) - b) / C_ATT)


def _enc_item(f) -> float:
    try:
        p = 4 * f.mm_position.get_num_embeds()
    except Exception:
        p = 4 * f.mm_position.length
    q = min(p, ENC_FRAME) if f.modality == "video" else p
    return p * (E_LIN + E_ATT * q)


def enc_items(request, start: int, end: int) -> list:
    """(offset, encoder seconds) of the images whose placeholders start in [start, end): encoded in that chunk."""
    if not (E_LIN or E_ATT) or request is None or not getattr(request, "mm_features", None):
        return []
    return [(f.mm_position.offset, _enc_item(f)) for f in request.mm_features if start <= f.mm_position.offset < end]


def enc_left(request) -> float:
    return sum(c for _, c in enc_items(request, request.num_computed_tokens, 1 << 62))


def prefill_left(request) -> int:
    """Prompt (or recompute) tokens still to prefill; 0 for a decoding request."""
    if request.num_output_placeholders:
        return 0
    left = request.num_tokens - request.num_computed_tokens
    return left if left > 1 else 0


def order_running(running: list) -> None:
    """In place: decodes first (stable), then prefills by least remaining prefill time."""
    def key(r):
        left = prefill_left(r)
        return (1, cost(r.num_computed_tokens, left) + enc_left(r)) if left else (0, 0.0)
    running.sort(key=key)


class StepPlan:
    def __init__(self, running: list, waiting_ready: bool, token_budget: int, urgent: bool = False):
        self.decoding = any(not prefill_left(r) for r in running)
        self.time_left = T_ADD if self.decoding else T_SOLO
        backlog = sum(cost(r.num_computed_tokens, prefill_left(r)) + enc_left(r) for r in running if prefill_left(r))
        self.carried_prefill = False
        self.hold = decode_starved(self.decoding, backlog) | over_prefill_share(self.decoding, backlog)   # decode-only step
        global _last_plan
        _last_plan = self
        self.log = None
        if _LOG:
            dec = [r for r in running if not prefill_left(r)]
            self.log = {"t": round(time.monotonic(), 5), "dec": len(dec), "dctx": sum(r.num_computed_tokens for r in dec),
                        "pf": [], "hold": bool(self.hold), "bl": round(backlog, 3)}
            _log_step(self.log)
        if T_URGENT > 0 and urgent:
            self.time_left = min(self.time_left, T_URGENT)
        self.waiting_ready = waiting_ready
        reserve = RESERVE if RESERVE > 0 else token_budget // 4
        self.running_tokens = max(token_budget - reserve, 0) if waiting_ready else token_budget
        self.running_time = self.time_left * (1 - RESERVE_FRAC) if waiting_ready else self.time_left
        self.running_prefilled = False

    def cap(self, start: int, n: int, running: bool, request=None) -> int:
        """Clip a prefill chunk of n tokens that starts at position `start`."""
        if self.hold:
            return 0
        t = min(self.time_left, self.running_time) if running else self.time_left
        first = not self.running_prefilled if running else True
        if running:
            n = min(n, self.running_tokens)
        m = tokens_for(start, t)
        items = sorted(enc_items(request, start, start + min(n, max(m, MIN_CHUNK if first else 0)) + LOOKAHEAD))
        if items:
            # images in the chunk: their encoder time comes out of the step's time first. An image that does not fit
            # ends the chunk just before its placeholder; one at the chunk's first token runs only in a step's first
            # chunk (it has to run sometime, and a fresh step is the cheapest place).
            spent, last = 0.0, None
            for off, c in items:
                if cost(start, off - start) + spent + c > t:
                    if off - LOOKAHEAD > start:
                        m = off - start - LOOKAHEAD
                        break
                    if not first:
                        return 0
                spent += c
                last = off
            else:
                m = max(tokens_for(start, max(t - spent, 0.0)), last - start - LOOKAHEAD + 1)
                if first:
                    m = max(m, MIN_CHUNK)
                n = min(n, m)
                if ALIGN and m >= ALIGN and start + m - m % ALIGN + LOOKAHEAD > last:
                    m -= m % ALIGN
                return max(min(n, m), 0)
        elif first:
            m = max(m, MIN_CHUNK)
        if ALIGN and m < n and m >= ALIGN:
            m -= m % ALIGN
        return max(min(n, m), 0)

    def charge(self, start: int, n: int, running: bool, request=None) -> None:
        if n > 0:
            self.carried_prefill = True
            if self.log is not None:
                self.log["pf"].append([start, n])
        c = cost(start, n) + sum(e for _, e in enc_items(request, start, start + n + LOOKAHEAD))
        self.time_left -= c
        if running:
            self.running_prefilled = True
            self.running_tokens -= n
            self.running_time -= c
