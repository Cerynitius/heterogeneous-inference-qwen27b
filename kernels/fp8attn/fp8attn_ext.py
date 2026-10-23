"""Loader for the FP8 paged extend attention extension (kernels/fp8attn/fp8_extend.cu, sm_120a).
ext(): the prefill build (8 warps, 128 rows / CTA). ext_verify(): the spec-verify decode build (FP8_NW=4: 64 rows / CTA,
one CTA holds the 8 tokens x 6 heads of a request x KV head; call it in graph-safe mode, max_seq_len 0)."""
import importlib.util
import os
_EXT = {}


def _load(name, defs):
    if name not in _EXT:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        inc = os.path.join(importlib.util.find_spec("nvidia").submodule_search_locations[0], "cu13", "include")
        _EXT[name] = load(name, [f"{here}/fp8_extend.cu"], extra_cflags=["-idirafter", inc],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a",
                                             "-Xcompiler=-idirafter" + inc, "--use_fast_math"] + defs, verbose=False)
    return _EXT[name]


def ext():
    return _load("qwen38_fp8_extend", [])


def ext_verify():
    return _load("qwen38_fp8_verify", ["-DFP8_NW=4"])
