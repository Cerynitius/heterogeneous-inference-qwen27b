"""JIT-builds gdn_defer.cu (sm_120a) with torch.utils.cpp_extension; `from gdn_defer_jit import ext` (gdn_mtp, gdn_materialize).
--use_fast_math as vLLM builds fused_gdn_decode_kernel.cu (CMakeLists v0.30.0); without it the outputs are not bit-identical."""
import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_flags = os.environ.get("GDN_DEFER_NVCC", "").split()   # testing only: extra nvcc flags
_name = "gdn_defer_ext" + ("_" + str(abs(hash(" ".join(_flags))) % 10**8) if _flags else "")
_bdir = os.path.join(HERE, "build" + ("_" + _name.split("_")[-1] if _flags else ""))
os.makedirs(_bdir, exist_ok=True)
ext = load(name=_name, sources=[os.path.join(HERE, "gdn_defer.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo", "--use_fast_math"] + _flags,
           build_directory=_bdir, verbose=False)
