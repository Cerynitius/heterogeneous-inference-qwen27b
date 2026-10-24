"""vLLM glue for gdn_defer: buffers, per-step epoch / spec-row map, kernel A per GDN layer, kernel B per step.

Wiring (hetero-qwen27b changes in vLLM 0.30, active with QWEN38_GDN_DEFER=1):
  GDNAttentionMetadataBuilder.__init__  -> init(...)          allocate (eager, before any graph capture)
  GDNAttentionMetadataBuilder.build     -> begin_step(mask, n) epoch += 1 on the device, spec row -> batch row map
  GDN layer spec decode (fused MTP path) -> decode(layer, ...) kernel A instead of the stock kernel
  GPUModelRunner._update_states_after_model_execute (V1 runner) and the mamba-hybrid model state of the V2 runner
  (worker/gpu/model_states/mamba_hybrid.py), right after the accepted counts and before the mamba postprocess
                                        -> materialize(...)    kernel B
All buffers have fixed addresses, so the forward can be a CUDA graph (kernel A reads the epoch from device memory).
"""
import os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_defer_jit import ext  # noqa: E402  (JIT build at import, never inside a graph capture)

_D = 128


class GdnDefer:
    def __init__(self, num_layers: int, max_rows: int, slots: int, num_k_heads: int, num_v_heads: int, device):
        L, R, S, H, HV = num_layers, max_rows, slots, num_k_heads, num_v_heads
        self.L, self.R, self.S, self.H, self.HV = L, R, S, H, HV
        self.k = torch.zeros(L, R, H, S, _D, device=device)
        self.v = torch.zeros(L, R, HV, S, _D, device=device, dtype=torch.bfloat16)
        self.g = torch.zeros(L, R, HV, S, 2, device=device)
        self.meta = torch.full((L, R, 3 + S), -1, device=device, dtype=torch.int32)
        self.epoch = torch.zeros(1, device=device, dtype=torch.int32)
        self.rowmap = torch.zeros(R, device=device, dtype=torch.int32)
        self._rowmap_cpu = torch.zeros(R, dtype=torch.int32, pin_memory=True)
        self.state_ptrs = torch.zeros(L, device=device, dtype=torch.int64)
        self._ptrs_host = [0] * L
        self._ptrs_dirty = True
        self.slot_stride = None
        self._layer_idx: dict[str, int] = {}
        self.steps = self.materialized = 0
        mib = sum(t.numel() * t.element_size() for t in (self.k, self.v, self.g, self.meta)) / 2**20
        print(f"[qwen38 gdn_defer] on: {L} GDN layers x {R} rows x {S} slots, buffers {mib:.0f} MiB", flush=True)

    def begin_step(self, spec_mask_cpu, num_spec: int) -> None:
        self.epoch.add_(1)
        if num_spec:
            rows = torch.nonzero(spec_mask_cpu, as_tuple=False).flatten()
            assert rows.numel() == num_spec <= self.R, (rows.numel(), num_spec, self.R)
            self._rowmap_cpu[:num_spec].copy_(rows)
            self.rowmap[:num_spec].copy_(self._rowmap_cpu[:num_spec], non_blocking=True)
            self.steps += 1

    def decode(self, prefix: str, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted, state,
               output_gate, norm_weight, out, scale, norm_eps, activation) -> None:
        i = self._layer_idx.get(prefix)
        if i is None:
            i = self._layer_idx[prefix] = len(self._layer_idx)
            assert i < self.L, f"more GDN layers than {self.L}"
        if self._ptrs_host[i] != state.data_ptr():
            self._ptrs_host[i] = state.data_ptr()
            self._ptrs_dirty = True
            assert self.slot_stride in (None, state.stride(0)), "GDN state slot strides differ between layers"
            self.slot_stride = state.stride(0)
        ext.gdn_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted, state, output_gate,
                    norm_weight, out, scale, norm_eps, activation == "sigmoid", True,
                    self.k[i], self.v[i], self.g[i], self.meta[i], self.epoch)

    def materialize(self, num_accepted_gpu, num_computed=None, num_scheduled=None, num_draft=None,
                    block_size: int = 0, idx_map=None, precomputed: bool = False) -> None:
        """V1 runner: per-batch-row counts. V2 runner: idx_map (batch row -> request slot) with per-slot counts and
        precomputed=True (num_computed = post-step count)."""
        if self.slot_stride is None:      # no deferred decode has run yet
            return
        if self._ptrs_dirty:
            assert len(self._layer_idx) == self.L, f"{len(self._layer_idx)} of {self.L} GDN layers registered"
            self.state_ptrs.copy_(torch.tensor(self._ptrs_host, dtype=torch.int64))
            self._ptrs_dirty = False
        ext.gdn_materialize(self.state_ptrs, self.slot_stride, self.k, self.v, self.g, self.meta, self.epoch,
                            self.rowmap, num_accepted_gpu, num_computed, num_scheduled, num_draft, block_size,
                            self.R, self.H, self.HV, idx_map, precomputed)   # kernel B skips rows not stamped now
        self.materialized += 1


DEFER: GdnDefer | None = None


def init(vllm_config, device) -> "GdnDefer | None":
    """Called from the GDN metadata builder; idempotent. None unless QWEN38_GDN_DEFER=1 and the config fits."""
    global DEFER
    if DEFER is not None or os.environ.get("QWEN38_GDN_DEFER", "0") != "1":
        return DEFER
    hf = vllm_config.model_config.hf_text_config
    spec = vllm_config.speculative_config
    if spec is None or vllm_config.parallel_config.pipeline_parallel_size != 1 \
            or vllm_config.parallel_config.tensor_parallel_size != 1:
        print("[qwen38 gdn_defer] off: needs spec decode, TP=1, PP=1", flush=True)
        return None
    layers = sum(1 for t in hf.layer_types if t == "linear_attention")
    DEFER = GdnDefer(layers, vllm_config.scheduler_config.max_num_seqs, spec.num_speculative_tokens + 1,
                     hf.linear_num_key_heads, hf.linear_num_value_heads, device)
    return DEFER
