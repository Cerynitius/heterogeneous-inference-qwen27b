"""Row argmax for the MTP draft head: replaces logits.max(dim=-1) (a 1-CTA-per-row reduce, ~23 us at
98304 columns on the PRO 5000) with a 2-stage split reduction. Same result as torch.max(dim=-1).indices: the largest
value, ties -> lowest index; NaN counts as the maximum (first NaN wins), as in torch (CUDA-graph padding rows can
hold NaN logits). Stage 1 keeps each chunk's first maximum / first NaN, stage 2 picks among chunks in index order."""
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _argmax_stage1(x_ptr, stride, n, vals_ptr, idx_ptr, nchunk, BLOCK: tl.constexpr):
    row = tl.program_id(0); c = tl.program_id(1)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + row * stride + offs, mask=offs < n, other=-float("inf")).to(tl.float32)
    isnan = v != v
    first_nan = tl.min(tl.where(isnan, offs, 2147483647), 0)
    vf = tl.where(isnan, -float("inf"), v)
    m = tl.max(vf, 0)
    i = tl.min(tl.where(vf == m, offs, 2147483647), 0)         # first index of the chunk maximum
    has_nan = first_nan < 2147483647
    tl.store(vals_ptr + row * nchunk + c, tl.where(has_nan, float("nan"), m))
    tl.store(idx_ptr + row * nchunk + c, tl.where(has_nan, first_nan, i))


@triton.jit
def _argmax_stage2(vals_ptr, idx_ptr, out_ptr, nchunk, CB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, CB)
    v = tl.load(vals_ptr + row * nchunk + offs, mask=offs < nchunk, other=-float("inf"))
    ix = tl.load(idx_ptr + row * nchunk + offs, mask=offs < nchunk, other=2147483647)
    isnan = v != v
    nan_ix = tl.min(tl.where(isnan, ix, 2147483647), 0)        # a chunk with NaN wins, the earliest one
    vf = tl.where(isnan, -float("inf"), v)
    m = tl.max(vf, 0)
    i = tl.min(tl.where(vf == m, ix, 2147483647), 0)           # chunks are in index order -> lowest index wins
    i = tl.where(nan_ix < 2147483647, nan_ix, i)
    tl.store(out_ptr + row, i.to(tl.int64))


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.stride(1) == 1
    M, n = x.shape
    nchunk = triton.cdiv(n, BLOCK)
    vals = torch.empty(M, nchunk, device=x.device, dtype=torch.float32)
    idx = torch.empty(M, nchunk, device=x.device, dtype=torch.int32)
    out = torch.empty(M, device=x.device, dtype=torch.int64)
    _argmax_stage1[(M, nchunk)](x, x.stride(0), n, vals, idx, nchunk, BLOCK=BLOCK, num_warps=8)
    _argmax_stage2[(M,)](vals, idx, out, nchunk, CB=triton.next_power_of_2(nchunk), num_warps=1)
    return out
