"""JIT-builds fp4_gemv.cu (sm_120a) with torch.utils.cpp_extension; `from fp4_gemv_jit import fp4_gemv`."""
import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_unroll = os.environ.get("FP4_GEMV_UNROLL", "")  # tuning only: phase-loop unroll override
_name = "fp4_gemv_ext" + (f"_u{_unroll}" if _unroll else "")
_bdir = os.path.join(HERE, "build" + (f"_u{_unroll}" if _unroll else ""))
os.makedirs(_bdir, exist_ok=True)
_mod = load(name=_name, sources=[os.path.join(HERE, "fp4_gemv.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo"]
            + ([f"-DFP4_GEMV_UNROLL={_unroll}"] if _unroll else []),
            build_directory=_bdir, verbose=False)
fp4_gemv = _mod.fp4_gemv
