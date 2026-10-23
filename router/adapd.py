#!/usr/bin/env python3
"""Adaptive prefill/decode router for the two-card Qwen3.8-27B setup.

Both vLLM servers run NixlConnector as kv_both, so each request can go one of two ways:
  * aggregated: prefill and decode on one card (any request; no KV transfer);
  * split: prefill on card P (kv_transfer_params do_remote_decode, max_tokens 1), then decode on card D, which pulls
    the KV (and GDN state) over NIXL; D reuses whatever prefix it already has cached and pulls only the rest.
Per request the router predicts prefill time on each card with the scheduler's cost model (sched/qwen38_sched.py,
per-card constants), minus the prefix the card is believed to hold:
  * short (best predicted prefill <= --short-s): aggregated on the card with the best score = predicted prefill +
    penalty if the card is running a long prefill (its decodes are throttled to T_ADD) + small load term. A large
    cache hit makes the prefill tiny on that card only, so follow-ups stay with their cache;
  * long: P = the card where the prefill finishes first, counting that decoding streams on a card slow prefill to
    the T_ADD share and pay for it with throttled tokens; D = the other card if it has a free seat and KV room,
    else P. Long prefills thus drift onto whichever card is not decoding and the new stream decodes elsewhere;
    with no long prompts both cards just serve aggregated traffic.
Which prefixes a card holds is tracked in the router (chained hashes every --hash-chars of the serialized messages,
LRU-capped by the card's KV token capacity): approximate, like vllm-router's cache_aware tree.
--home-split: as --home, but a session's first prefill may run on the other card and reach its home over NIXL.
--home (agent sessions): a session (= its first user message) lives on one card. Its first request picks the card
that fits it (KV of the sessions already living there + this prompt <= --kv-fill of the card's KV tokens, seats free)
with the lowest decode attention load per step afterwards, sum(session tokens) * C_ATT; ties -> earliest prefill
finish. Later requests of the session stay on that card (aggregated), so its history never has to be re-prefilled.
Sessions idle for --session-ttl s (default 300) stop counting.

  adapd.py --port 8090 --backend w6=http://127.0.0.1:8000,114,2.36,6,1382480 \
                       --backend w5=http://127.0.0.1:8001,127,3.06,6,859656
  (name=url,C_LIN_us,C_ATT_ns,max_num_seqs,kv_tokens)   GET /router/state shows the live view.
"""
import argparse, asyncio, hashlib, itertools, json, logging, math, os, re, time, uuid
from collections import OrderedDict

import aiohttp
from aiohttp import web

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--backend", action="append", required=True)
ap.add_argument("--short-s", type=float, default=2.0, help="prefill seconds at or below which a request stays aggregated")
ap.add_argument("--tadd-share", type=float, default=0.57, help="prefill speed share on a card with decodes (T_ADD 40 ms)")
ap.add_argument("--stall-w", type=float, default=0.05, help="seconds of score per decoding stream per prefill second")
ap.add_argument("--hash-chars", type=int, default=2048)
ap.add_argument("--tokenize-above", type=int, default=32768, help="exact /tokenize count above this many chars")
ap.add_argument("--no-split", action="store_true", help="never split (aggregated-only cost-model routing)")
ap.add_argument("--home", action="store_true", help="session homing with per-card KV accounting (agent sessions)")
ap.add_argument("--kv-fill", type=float, default=0.98, help="--home: fill limit of a card's KV tokens")
ap.add_argument("--home-split", action="store_true",
                help="--home + adaptive P/D for a session's first request: home D as with --home, P = the card "
                     "that finishes the prefill first; split when P != D and the prefill is longer than --short-s")
ap.add_argument("--session-ttl", type=float, default=300.0)
ap.add_argument("--no-reuse-ids", dest="reuse_ids", action="store_false",
                help="do not forward P's prompt token ids to D (D re-applies the chat template and re-tokenizes)")
ap.add_argument("--log", default="")
ap.add_argument("--ctx", type=int, default=262144, help="max_model_len of the servers")
ap.add_argument("--think-caps", default="low=49152,medium=65536,high=98304,xhigh=131072,max=0",
                help="hard thinking budget per effort tier (0 = whatever the context leaves after the answer room); "
                     "a safety net only, never hit naturally: >= 2x the longest uncapped thinking measured per "
                     "template level (low 23.7K, medium 15.4K, xhigh 39.1K; p2p/think_tiers.py --parts natural)")
ap.add_argument("--answer-min", type=int, default=16384, help="visible-answer room thinking can never take")
ap.add_argument("--compact-trigger", type=int, default=-1,
                help="compact a chat session once its prompt passes this many tokens; -1 (default) = "
                     "only when it no longer fits: prompt > ctx - --answer-min (would be a 400 / no answer room), 0 = off. "
                     "Like the industry default the full 256K stays verbatim; near the end thinking room shrinks")
ap.add_argument("--compact-keep", type=int, default=24000, help="recent tokens kept verbatim after a compaction")
ap.add_argument("--compact-head", type=int, default=8192, help="the first user message stays verbatim up to this size")
ap.add_argument("--img-max-pixels", type=int, default=4194304,
                help="images above this many pixels are shrunk in the router (imgshrink.py); 0 = off")
ap.add_argument("--img-workers", type=int, default=3, help="image shrink worker processes")
ap.add_argument("--img-cache-mb", type=int, default=1024, help="cache of shrunk images (by input hash)")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    filename=args.log or None)
log = logging.getLogger("adapd")
from imgshrink import Shrinker
SHRINK = Shrinker(args.img_max_pixels, args.img_workers, args.img_cache_mb << 20, 64 << 20, log)
XFER_GBPS = 13.0          # NIXL push measured (96K in 0.26 s); only logged (info xfer_s)
KV_BYTES_PER_TOKEN = 37000   # measured: FP8 KV + GDN/conv states + DFlash2 drafter KV, 5.6 GB / 153K tok


class Backend:
    def __init__(self, spec):
        name, rest = spec.split("=", 1)
        url, clin, catt, seqs, kvtok = rest.split(",")
        self.name, self.url = name, url.rstrip("/")
        self.c_lin, self.c_att = float(clin) * 1e-6, float(catt) * 1e-9
        self.max_seqs, self.kv_tokens = int(seqs), int(kvtok)
        self.reqs = {}                     # rid -> Track
        self.prefix = OrderedDict()        # chained hash -> chars covered (LRU)
        self.prefix_cap = max(64, int(self.kv_tokens * 3.0 / args.hash_chars))
        self.sessions = {}                 # --home: session key -> [tokens, last use]

    def resident(self, now):
        for k in [k for k, v in self.sessions.items() if now - v[1] > args.session_ttl]:
            del self.sessions[k]
        return sum(v[0] for v in self.sessions.values())

    def cost(self, start, n):
        return n * (self.c_lin + self.c_att * (start + n / 2)) if n > 0 else 0.0

    def decoding(self):
        return sum(1 for t in self.reqs.values() if t.phase == "decode")

    def long_prefill_left(self, now):
        return sum(max(t.pred - (now - t.t0), 0.1) for t in self.reqs.values()
                   if t.phase == "prefill" and t.pred > args.short_s)

    def tokens_in_use(self):
        return sum(t.n for t in self.reqs.values())

    def hit_chars(self, hashes):
        best = 0
        for h, c in hashes:
            if h in self.prefix:
                best = c
            else:
                break
        return best

    def remember(self, hashes):
        for h, c in hashes:
            self.prefix[h] = c
            self.prefix.move_to_end(h)
        while len(self.prefix) > self.prefix_cap:
            self.prefix.popitem(last=False)


class Track:
    def __init__(self, n, pred, phase="prefill"):
        self.n, self.pred, self.phase, self.t0 = n, pred, phase, time.time()


B = [Backend(s) for s in args.backend]
rid_counter = itertools.count()
cpt = [3.0]                                # chars per token (EMA from exact counts)
session: aiohttp.ClientSession | None = None


def serialize(body):
    if "messages" in body:
        parts = []
        for m in body["messages"]:
            c = m.get("content")
            parts.append(f"{m.get('role')}\x00{c if isinstance(c, str) else json.dumps(c, sort_keys=True)}\x01")
        return "".join(parts)
    p = body.get("prompt", "")
    return p if isinstance(p, str) else json.dumps(p)


def chained_hashes(text):
    out, h = [], hashlib.blake2b(digest_size=16)
    for end in range(args.hash_chars, len(text) + 1, args.hash_chars):
        h.update(text[end - args.hash_chars:end].encode("utf-8", "surrogatepass"))
        out.append((h.copy().hexdigest(), end))
    return out


def estimate(body, text):
    """Token estimate; image parts count as their model tokens, not as base64 characters."""
    ic, it = SHRINK.measure(body)
    return int((len(text) - ic) / cpt[0]) + it + 16


def text_only(msgs):
    return [dict(m, content=[p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "image_url")])
            if isinstance(m.get("content"), list) else m for m in msgs]


async def count_tokens(body, text):
    ic, it = SHRINK.measure(body)
    chars = len(text) - ic
    if chars < args.tokenize_above:
        return int(chars / cpt[0]) + it + 16
    req = {"model": body.get("model")}
    if "messages" in body:
        req.update(messages=text_only(body["messages"]) if ic else body["messages"], add_generation_prompt=True)
        if body.get("tools"):
            req["tools"] = body["tools"]
    else:
        req["prompt"] = body.get("prompt", "")
    try:
        async with session.post(B[0].url + "/tokenize", json=req, timeout=aiohttp.ClientTimeout(total=120)) as r:
            n = (await r.json())["count"]
        cpt[0] = 0.8 * cpt[0] + 0.2 * (chars / max(n, 1))
        return n + it
    except Exception as e:
        log.info(f"tokenize failed ({type(e).__name__}: {e}); estimating")
        return int(chars / cpt[0]) + it + 16


def session_key(body):
    for m in body.get("messages", []):
        if m.get("role") == "user":
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c, sort_keys=True)
            return hashlib.blake2b(f"{len(c)}\x00{c[:8192]}".encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
    # raw /v1/completions prompt: the first user turn of the chat-templated text keys the session
    p = body.get("prompt")
    if isinstance(p, str):
        i = p.find("<|im_start|>user")
        if i >= 0:
            return hashlib.blake2b(("raw\x00" + p[i:i + 8192]).encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
    return None


def decide_home(n, hashes, skey):
    now = time.time()
    info = {"n": n, "home": True, "hit": {b.name: min(int(b.hit_chars(hashes) / cpt[0]), max(n - 1, 0)) for b in B}}
    for b in B:
        if skey in b.sessions:
            b.sessions[skey] = [n, now]
            info["stay"] = b.name
            return b, b, info
    rows = []
    for b in B:
        res = b.resident(now)
        hit = info["hit"][b.name]
        fits = res + n <= args.kv_fill * b.kv_tokens and len(b.sessions) < b.max_seqs
        finish = b.long_prefill_left(now) + b.cost(hit, n - hit) / (args.tadd_share if b.decoding() else 1.0)
        rows.append((not fits, (res + n) * b.c_att, finish, b))
        info[b.name] = {"res": res, "fits": fits, "load_ms": round((res + n) * b.c_att * 1e3, 2), "finish": round(finish, 1)}
    b = min(rows, key=lambda r: (r[0], round(r[1] * 1e3, 1), r[2]))[3]
    b.sessions[skey] = [n, now]
    if args.home_split:
        # The session's history will live on b (D). Its first prefill may run on the other card if that one finishes
        # it first (it is not decoding, or b is busy with a long prefill); D then receives the KV over NIXL and keeps
        # it as the session's prefix cache, so later (incremental) turns stay aggregated on D.
        def p_finish(c):
            hit = info["hit"][c.name]
            return c.long_prefill_left(now) + c.cost(hit, n - hit) / (args.tadd_share if c.decoding() else 1.0)
        p = min(B, key=lambda c: (len(c.reqs) >= c.max_seqs, p_finish(c)))
        info["p_finish"] = {c.name: round(p_finish(c), 1) for c in B}
        if p is not b and p_finish(b) > args.short_s and len(p.reqs) < p.max_seqs:
            info["xfer_s"] = round(n * KV_BYTES_PER_TOKEN / (XFER_GBPS * 1e9), 2)
            return p, b, info
    return b, b, info


def decide(n, hashes):
    """Returns (P, D, info). P is D for aggregated requests."""
    now = time.time()
    rows = []
    for b in B:
        hit = min(int(b.hit_chars(hashes) / cpt[0]), max(n - 1, 0))
        rows.append((b, hit, b.cost(hit, n - hit)))
    full = {b.name: len(b.reqs) >= b.max_seqs or b.tokens_in_use() + n > 0.95 * b.kv_tokens for b in B}
    info = {"n": n, "hit": {b.name: h for b, h, _ in rows}, "cost": {b.name: round(c, 2) for b, _, c in rows}}
    if args.no_split or min(c for _, _, c in rows) <= args.short_s:
        def score(r):
            b, _, c = r
            return (full[b.name], c + (0.5 if b.long_prefill_left(now) > 0 else 0.0) + 0.05 * len(b.reqs))
        b = min(rows, key=score)[0]
        return b, b, info

    def p_score(r):
        b, _, c = r
        dec = b.decoding()
        t = b.long_prefill_left(now) + c / (args.tadd_share if dec else 1.0)
        return (full[b.name], t + args.stall_w * dec * t)
    p = min(rows, key=p_score)[0]
    others = [b for b in B if b is not p and not full[b.name]]
    d = min(others, key=lambda b: (b.decoding(), len(b.reqs))) if others else p
    info["xfer_s"] = round(n * KV_BYTES_PER_TOKEN / (XFER_GBPS * 1e9), 2)
    return p, d, info


# ---- chain-of-thought privacy ----
# Requests: assistant turns lose reasoning_content / reasoning and any <think>...</think> in their text, chat requests get
# include_reasoning=False (vLLM drops reasoning deltas and token ids / logprobs of reasoning) and preserve_thinking=False
# (the template renders no earlier thinking). Raw /v1/completions prompts lose closed <think> blocks before the final
# assistant turn; their output is cut to what follows </think> (the generation prompt opens <think>).
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


# Thinking provisioning, Claude API semantics on top of vLLM:
#   effort tiers low / medium / high / xhigh (default) / max from output_config.effort, reasoning_effort or
#   chat_template_kwargs.reasoning_effort; thinking {type: adaptive | enabled + budget_tokens | disabled} (or vLLM's
#   thinking_token_budget) sets an explicit budget. Each tier = the template's soft instruction + a hard cap
#   (vLLM thinking_token_budget: at the cap the server forces the end-of-thinking string and the model answers).
#   The answer room is reserved first: thinking <= ctx - prompt - --answer-min; the visible answer gets
#   room = max(client max_tokens, --answer-min) (capped by the context left) on top of the budget, server max_tokens =
#   budget + room (or unset = the whole context left), so thinking can never eat the visible answer.
TIER_TEMPLATE = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
THINK_CAPS = {k: int(v) for k, v in (kv.split("=") for kv in args.think_caps.split(","))}


def fit_max_tokens(body, n, chat):
    """A client max_tokens at or past the context end (prompt counted without tools, ...) would be a 400: chat drops
    it (vLLM's default is exactly the context left), completions get the safe remainder."""
    safe = args.ctx - n - n // 10 - 1024
    want = body.get("max_tokens") or body.get("max_completion_tokens")
    if want and int(want) >= safe:
        body.pop("max_completion_tokens", None)
        if chat:
            body.pop("max_tokens", None)
        else:
            body["max_tokens"] = max(safe, 1)


def provision(body, n, think_raw):
    """In place, after scrub_request. Returns a short description for the log."""
    ctk = dict(body.get("chat_template_kwargs") or {})
    th = body.pop("thinking", None) or {}
    oc = body.pop("output_config", None) or {}
    effort = oc.get("effort") or body.pop("reasoning_effort", None) or ctk.get("reasoning_effort") or "xhigh"
    if effort not in TIER_TEMPLATE:
        effort = "xhigh"
    budget = body.pop("thinking_token_budget", None)
    if th.get("type") == "enabled" and th.get("budget_tokens") is not None:
        budget = int(th["budget_tokens"])
    chat = "messages" in body
    if chat and (th.get("type") == "disabled" or ctk.get("enable_thinking") is False or budget == 0):
        ctk["enable_thinking"] = False
        ctk.pop("reasoning_effort", None)
        body["chat_template_kwargs"] = ctk
        fit_max_tokens(body, n, chat)
        return "thinking off"
    if not chat and not think_raw:
        fit_max_tokens(body, n, chat)
        return "no thinking"
    if chat:
        ctk["reasoning_effort"] = TIER_TEMPLATE[effort]
        body["chat_template_kwargs"] = ctk
    want = body.pop("max_completion_tokens", None) or body.get("max_tokens")
    left = args.ctx - n
    cap = budget if budget is not None else (THINK_CAPS.get(effort) or left)
    cap = max(1, min(cap, max(left - args.answer_min, min(1024, left // 4))))   # vLLM does not enforce 0
    body["thinking_token_budget"] = cap
    room = min(max(int(want or left), args.answer_min), left - cap)
    # client max_tokens = answer room (thinking comes on top). n is only estimated for homed sessions and vLLM rejects
    # max_tokens beyond the context left (400), so near the end of the context it is dropped: vLLM's default is
    # exactly what the context leaves
    safe = left - n // 10 - 1024
    if cap + room < safe:
        body["max_tokens"] = cap + room
    elif chat:
        body.pop("max_tokens", None)
        room = left - cap
    else:                                        # /v1/completions defaults to 16 tokens: send the safe remainder
        body["max_tokens"] = max(safe, cap + 1)
        room = body["max_tokens"] - cap
    return f"effort {effort} budget {cap} room {room}"


# Compaction, Claude's threshold compaction (compact_20260112) done in the router, so any
# OpenAI-style client gets it. Default: only a session that no longer fits is compacted (prompt >
# ctx - answer room, which would otherwise be a 400); clients compact earlier themselves or opt in per request.
# When a chat session's prompt passes the trigger, the middle of the history (after
# the system prompt and a short first user message, before the last ~--compact-keep tokens; the kept tail starts at a
# user or assistant message, never inside a tool round) is summarized by the model on the session's card (same tools /
# template arguments, so the summary request hits the session's prefix cache) and replaced by one user message holding
# the summary. Clients keep sending their full history: the router recognizes the compacted prefix by its hash and
# swaps in the same replacement every time (deterministic -> the rewritten prompt stays prefix-cached), compacting
# again when the rewritten prompt passes the trigger. Chain of thought never enters the summary (scrubbed before).
# Per request: context_management {edits: [{type: compact_20260112, trigger: {value: N}, instructions: "..."}]}
# overrides trigger / summary prompt, edits: [] turns it off. The response carries X-Context-Compacted: before->after.
COMPACT_PROMPT = ("The conversation above has grown too long to continue as is and will be replaced by a summary that you "
                  "will continue from. Write that summary now. Keep everything needed to carry on the work seamlessly: the "
                  "user's goal, requirements and constraints; what has been done and decided (files, commands, code changes, "
                  "values, results, tool outputs that still matter, verbatim where exactness matters); the current state; open "
                  "problems, errors and next steps. Be complete on state and facts, brief on narrative. Do not call tools. "
                  "Output only the summary inside <summary></summary>.")
COMPACTIONS = OrderedDict()                # (skey, k original messages, hash of them) -> replacement messages
COMPACT_LOCKS = {}


def msg_hash(msgs):
    return hashlib.blake2b(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode("utf-8", "surrogatepass"),
                           digest_size=16).hexdigest()


def msg_tokens(m):
    ic, it = SHRINK.measure({"messages": [m]})
    c = m.get("content")
    c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return int((len(c or "") - ic + len(json.dumps(m.get("tool_calls") or "", ensure_ascii=False))) / cpt[0]) + it + 8


def compact_config(body):
    cm = body.pop("context_management", None)
    trig, instr = (args.ctx - args.answer_min if args.compact_trigger < 0 else args.compact_trigger), COMPACT_PROMPT
    if isinstance(cm, dict) and "edits" in cm:
        ed = [e for e in cm["edits"] or [] if str(e.get("type", "")).startswith("compact")]
        if not ed:
            return 0, instr
        trig = int((ed[0].get("trigger") or {}).get("value") or trig)
        instr = ed[0].get("instructions") or instr
    return trig, instr


def apply_compaction(body, skey):
    """Swap the longest stored compacted prefix of this session for its summary. Returns (k original, offset)."""
    msgs = body["messages"]
    for (sk, k, h), repl in reversed(list(COMPACTIONS.items())):
        if sk == skey and len(msgs) > k and msg_hash(msgs[:k]) == h:
            COMPACTIONS.move_to_end((sk, k, h))
            body["messages"] = list(repl) + msgs[k:]
            return k, len(repl) - k
    return 0, 0


async def compact(body, skey, n, instr, orig, off):
    """Summarize the middle of body["messages"] (already rewritten by apply_compaction: original index = i - off).
    Returns a log string; body is rewritten in place on success."""
    msgs = body["messages"]
    h = 0
    while h < len(msgs) and msgs[h].get("role") == "system":
        h += 1
    if h < len(msgs) and msgs[h].get("role") == "user" and msg_tokens(msgs[h]) <= args.compact_head:
        h += 1
    # a session already past the context: keep more verbatim so the summary request itself fits (>= 32K spare)
    keep = max(args.compact_keep, n - (args.ctx - 32000))
    i, tail = len(msgs), 0
    while i - 1 > h and tail + msg_tokens(msgs[i - 1]) <= keep:
        i -= 1; tail += msg_tokens(msgs[i])
    while i < len(msgs) and msgs[i].get("role") not in ("user", "assistant"):
        i += 1                               # the tail never starts inside a tool round
    if i >= len(msgs):                       # the last turn alone is longer than the keep size: keep just it
        i = len(msgs) - 1
        while i > h and msgs[i].get("role") not in ("user", "assistant"):
            i -= 1
    if i <= h:
        return "compaction: nothing before the kept tail"
    sbody = {k: v for k, v in body.items() if k in ("model", "tools", "chat_template_kwargs", "include_reasoning")}
    ctk = dict(sbody.get("chat_template_kwargs") or {})
    effort = effort_of(body, ctk)
    ctk["reasoning_effort"] = TIER_TEMPLATE[effort]   # the session's own template arguments: prefix-cache hit
    n_s = n - tail                           # the summary request's prompt; max_tokens unset = the whole context left
    sbody.update(messages=msgs[:i] + [{"role": "user", "content": instr}], chat_template_kwargs=ctk, stream=False,
                 thinking_token_budget=max(1024, min(THINK_CAPS.get("medium", 65536), args.ctx - n_s - 24000)),
                 include_reasoning=False)
    if sbody.get("tools"):
        sbody["tool_choice"] = "none"
    b = next((b for b in B if skey in b.sessions), None) or min(B, key=lambda b: len(b.reqs))
    t0 = time.time()
    tr = Track(n, b.cost(int(0.9 * n), n - int(0.9 * n)))
    b.reqs[f"compact-{skey}"] = tr
    try:
        async with session.post(b.url + "/v1/chat/completions", json=sbody,
                                timeout=aiohttp.ClientTimeout(total=None, sock_read=3600)) as r:
            js = await r.json()
            if r.status != 200:
                return f"compaction failed on {b.name}: HTTP {r.status} {json.dumps(js)[:300]}"
    except Exception as e:
        return f"compaction failed on {b.name}: {type(e).__name__}: {e}"
    finally:
        b.reqs.pop(f"compact-{skey}", None)
    text = (js["choices"][0]["message"].get("content") or "").strip()
    m = re.search(r"<summary>(.*?)(</summary>|$)", text, re.S)
    text = (m.group(1) if m else text).strip()
    if js["choices"][0].get("finish_reason") != "stop" or not text:
        return f"compaction failed on {b.name}: finish {js['choices'][0].get('finish_reason')}, {len(text)} ch of summary"
    repl = msgs[:h] + [{"role": "user", "content": "[Context compacted: the earlier part of this conversation was replaced "
                                                  f"by this summary]\n<summary>\n{text}\n</summary>"}]
    k = i - off                              # in the client's original messages
    COMPACTIONS[(skey, k, msg_hash(orig[:k]))] = repl
    while len(COMPACTIONS) > 512:
        COMPACTIONS.popitem(last=False)
    body["messages"] = repl + msgs[i:]
    return (f"compacted messages {h}..{i - 1} ({i - h} msgs) on {b.name} in {time.time() - t0:.0f} s, summary "
            f"{js['usage']['completion_tokens']} tok (incl. thinking)")


def effort_of(body, ctk):
    oc = body.get("output_config") or {}
    e = oc.get("effort") or body.get("reasoning_effort") or ctk.get("reasoning_effort") or "xhigh"
    return e if e in TIER_TEMPLATE else "xhigh"


def scrub_request(body, path):
    """In place. Returns True when a raw completion opens in thinking (its output must be filtered)."""
    if "messages" in body:
        for m in body["messages"]:
            if m.get("role") != "assistant":
                continue
            m.pop("reasoning_content", None); m.pop("reasoning", None)
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = THINK_RE.sub("", c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = THINK_RE.sub("", part["text"])
        body["include_reasoning"] = False
        body["chat_template_kwargs"] = dict(body.get("chat_template_kwargs") or {}, preserve_thinking=False)
        return False
    p = body.get("prompt")
    if path.endswith("/completions") and isinstance(p, str):
        i = p.rfind("<|im_start|>assistant")
        head, tail = (p[:i], p[i:]) if i >= 0 else (p, "")
        body["prompt"] = THINK_RE.sub("", head) + tail
        return tail.rstrip().endswith("<think>")
    return False


class ThinkFilter:
    """Per choice: drop completion text up to and including </think> (and the blank line after it)."""
    def __init__(self):
        self.state = {}                      # choice index -> [in_think, carry]

    def text(self, idx, t):
        st = self.state.setdefault(idx, [True, ""])
        if not st[0]:
            return t
        buf = st[1] + t
        j = buf.find("</think>")
        if j < 0:
            st[1] = buf[-8:]                # keep a partial tag across chunks
            return ""
        st[0] = False
        return buf[j + len("</think>"):].lstrip("\n")

    def choice(self, c):
        idx = c.get("index", 0)
        in_think = self.state.get(idx, [True])[0]
        if "text" in c and isinstance(c["text"], str):
            c["text"] = self.text(idx, c["text"])
        if in_think:                         # no token ids / logprobs of hidden reasoning
            c.pop("token_ids", None); c.pop("logprobs", None)
        return c


def filter_completion(data, flt):
    try:
        js = json.loads(data)
    except Exception:
        return data
    for c in js.get("choices") or []:
        flt.choice(c)
    return json.dumps(js).encode()


async def relay(request, d, body, track, rid, t_in, headers=None, think=False, compacted=""):
    """Stream (or return) D's answer to the client; the track flips to decode at the first chunk."""
    stream = bool(body.get("stream"))
    async with session.post(d.url + request.path, json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=None, sock_read=3600)) as r:
        if not stream or r.status != 200:
            data = await r.read()
            track.phase = "decode"
            if think and r.status == 200:
                data = filter_completion(data, ThinkFilter())
            return web.Response(body=data, status=r.status, content_type=r.content_type,
                                headers={"X-Context-Compacted": compacted} if compacted else None)
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                                       **({"X-Context-Compacted": compacted} if compacted else {})})
        await resp.prepare(request)
        first = True
        flt, pend = (ThinkFilter(), b"") if think else (None, b"")
        async for chunk in r.content.iter_any():
            if first:
                first = False
                track.phase = "decode"
                log.info(f"{rid} first chunk {time.time() - t_in:.2f}s on {d.name}")
            if flt is None:
                await resp.write(chunk)
                continue
            pend += chunk                    # SSE events end with a blank line; rewrite whole events only
            *events, pend = pend.split(b"\n\n")
            out = []
            for ev in events:
                if ev.startswith(b"data: ") and ev[6:].strip() != b"[DONE]":
                    ev = b"data: " + filter_completion(ev[6:], flt)
                out.append(ev + b"\n\n")
            if out:
                await resp.write(b"".join(out))
        if pend:
            await resp.write(pend)
        await resp.write_eof()
        return resp


async def handle_gen(request):
    t_in = time.time()
    body = await request.json()
    think = scrub_request(body, request.path)
    await SHRINK.apply(body, session)
    rid = f"q{next(rid_counter)}"
    skey = session_key(body) if args.home or "messages" in body else None
    cinfo = ""
    if "messages" in body:
        trig, instr = compact_config(body)
        if trig:
            orig = list(body["messages"])        # the client's history: compactions are keyed on its prefixes
            _, off = apply_compaction(body, skey)
            if estimate(body, serialize(body)) > trig:
                async with COMPACT_LOCKS.setdefault(skey, asyncio.Lock()):
                    body["messages"] = list(orig)          # another request of the session may have compacted meanwhile
                    _, off = apply_compaction(body, skey)
                    n0 = await count_tokens(body, serialize(body))
                    if n0 > trig:
                        msg = await compact(body, skey, n0, instr, orig, off)
                        n1 = estimate(body, serialize(body))
                        cinfo = f"{n0}->{n1}" if msg.startswith("compacted") else ""
                        log.info(f"{rid} {msg} ({n0} -> ~{n1} tok)")
    text = serialize(body)
    hashes = chained_hashes(text)
    if skey and any(skey in b.sessions for b in B):   # homed: the count only updates its KV share, an estimate will do
        n = estimate(body, text)
    else:
        n = await count_tokens(body, text)
    prov = provision(body, n, think)
    p, d, info = decide_home(n, hashes, skey) if args.home else decide(n, hashes)
    split = p is not d
    log.info(f"{rid} {'split ' + p.name + '->' + d.name if split else 'agg ' + d.name} [{prov}] {json.dumps(info)}")
    tp = Track(n, p.cost(info["hit"][p.name], n - info["hit"][p.name]))
    p.reqs[rid] = tp
    try:
        if split:
            # same X-Request-Id on P and D: NixlPushConnector pairs D's registration with P's finished blocks by request id
            xid = {"X-Request-Id": uuid.uuid4().hex}
            pbody = dict(body)
            for k in ("stream", "stream_options", "max_completion_tokens"):
                pbody.pop(k, None)
            # return_token_ids: P's prompt ids go to D in kv_transfer_params["prompt_token_ids"] (vLLM's decode-side token
            # reuse, renderers/online_renderer.py), so D skips the chat template + tokenizer (0.9-1.4 s at 64-112K)
            pbody.update(max_tokens=1, min_tokens=0, return_token_ids=True, kv_transfer_params={
                "do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                "remote_block_ids": None, "remote_host": None, "remote_port": None})
            try:
                async with session.post(p.url + request.path, json=pbody, headers=xid,
                                        timeout=aiohttp.ClientTimeout(total=None, sock_read=7200)) as r:
                    pres = await r.json()
                    ok = r.status == 200 and pres.get("kv_transfer_params")
            except Exception as e:
                pres, ok = {"error": f"{type(e).__name__}: {e}"}, False
            p.remember(hashes)
            del p.reqs[rid]
            if not ok:   # prefill side failed: serve it aggregated on D instead
                log.info(f"{rid} split prefill on {p.name} failed: {json.dumps(pres)[:300]}; aggregated on {d.name}")
                dbody = body
            else:
                log.info(f"{rid} prefill done on {p.name} {time.time() - t_in:.2f}s")
                kvp = dict(pres["kv_transfer_params"])
                if args.reuse_ids and pres.get("prompt_token_ids"):
                    kvp["prompt_token_ids"] = pres["prompt_token_ids"]
                dbody = dict(body, kv_transfer_params=kvp)
            td = Track(n, 0.0 if ok else d.cost(0, n))
            d.reqs[rid] = td
            try:
                return await relay(request, d, dbody, td, rid, t_in, xid, think=think, compacted=cinfo)
            finally:
                d.reqs.pop(rid, None)
                d.remember(hashes)
        try:
            return await relay(request, d, body, tp, rid, t_in, think=think, compacted=cinfo)
        finally:
            d.remember(hashes)
    finally:
        p.reqs.pop(rid, None)
        log.info(f"{rid} done {time.time() - t_in:.1f}s")


async def passthrough(request):
    data = await request.read()
    async with session.request(request.method, B[0].url + request.path_qs, data=data,
                               headers={"Content-Type": request.headers.get("Content-Type", "application/json")}) as r:
        return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)


async def state(request):
    now = time.time()
    return web.json_response({"chars_per_token": round(cpt[0], 3), "backends": [{
        "name": b.name, "reqs": {k: [t.phase, t.n, round(t.pred, 1), round(now - t.t0, 1)] for k, t in b.reqs.items()},
        "decoding": b.decoding(), "long_prefill_left_s": round(b.long_prefill_left(now), 1),
        "sessions": len(b.sessions), "resident_tokens": b.resident(now),
        "prefix_entries": len(b.prefix)} for b in B]})


async def on_startup(app):
    global session
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))


async def on_cleanup(app):
    await session.close()
    if SHRINK.pool is not None:
        SHRINK.pool.shutdown(wait=False, cancel_futures=True)


app = web.Application(client_max_size=1 << 30)
app.router.add_post("/v1/chat/completions", handle_gen)
app.router.add_post("/v1/completions", handle_gen)
app.router.add_get("/router/state", state)
app.router.add_route("*", "/{tail:.*}", passthrough)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
# handler_cancellation: a client that disconnects while its prefill runs on P cancels the P request too
web.run_app(app, host=args.host, port=args.port, access_log=None, handler_cancellation=True)
