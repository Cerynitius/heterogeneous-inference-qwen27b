// W4A4 NVFP4 GEMV for small M (decode / MTP verify), RTX PRO 6000 (sm_120a)..
//
// out[t, n] = alpha * sum_g (sa[t,g] * sw[n,g]) * sum_{i<16} qa[t,16g+i] * qw[n,16g+i]
// Drop-in for flashinfer mm_fp4 (vLLM's NVFP4 linear) at M <= 8: same inputs (vLLM-quantized activations with
// 128x4-swizzled e4m3 block scales, checkpoint weights with swizzled scales, alpha = input_scale * weight_scale_2).
//
// Numerics = block-scaled MMA semantics, exact per group: e2m1 values decode exactly to f16; HFMA2 accumulates the
// even and the odd elements of a group separately, 8 products each, |partial| <= 8*36 = 288 in steps of 1/4, which
// f16 holds exactly (ulp 1/4 below 512), so every fma is exact. The two partials add exactly in fp32 (<= 576), the
// group scale product sa*sw is exact (4+4 significand bits) and so is group_sum * (sa*sw) (<= 20 bits); only the
// cross-group fp32 accumulation order differs from the tensor-core path.
//
// Work split follows NInfer's nvfp4_a16_gemv (github.com/Neroued/ninfer, Apache-2.0): 16 output rows per CTA
// (8 warps x 2 rows), lanes stride K in 16-value groups (one NVFP4 group per lane per phase), the CTA's rows are
// the 4 row quartets {q, q+32, q+64, q+96} of the swizzled scale layout so its weight scales are 64 contiguous bytes
// per 64-column tile, staged once into shared memory.
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: it pulls cusparse.h, absent here
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {
#ifndef FP4_GEMV_UNROLL
#define FP4_GEMV_UNROLL 4
#endif
constexpr int kUnroll = FP4_GEMV_UNROLL;
// CTA = kBlockRows rows = kBlockRows/4 swizzle quartets (8 or 16); warps x rows-per-warp = kBlockRows.

__device__ __forceinline__ float e4m3(std::uint8_t v) {
    __nv_fp8_e4m3 s;
    s.__x = v;
    return static_cast<float>(s);
}

// 16 e2m1 values -> 8 f16x2 (exact). Weights and activations use the same decode, so the pairing order inside a
// group is the same on both sides.
__device__ __forceinline__ void decode16(uint2 packed, std::uint32_t (&h)[8]) {
#pragma unroll
    for (int w = 0; w < 2; ++w)
        asm("{\n.reg .b8 b0, b1, b2, b3;\n"
            "mov.b32 {b0, b1, b2, b3}, %4;\n"
            "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
            "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
            : "=r"(h[4 * w]), "=r"(h[4 * w + 1]), "=r"(h[4 * w + 2]), "=r"(h[4 * w + 3])
            : "r"(w ? packed.y : packed.x));
}

__device__ __forceinline__ std::uint32_t hfma2(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    std::uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float group_sum(std::uint32_t h) {
    float lo, hi;
    asm("{\n.reg .f16 l, u;\nmov.b32 {l, u}, %2;\ncvt.f32.f16 %0, l;\ncvt.f32.f16 %1, u;\n}" : "=f"(lo), "=f"(hi) : "r"(h));
    return lo + hi;
}

// Swizzled 128x4 block-scale offset of (row, group) for a matrix with `groups` scale columns.
__device__ __forceinline__ std::int64_t sf_offset(int row, int group, int groups) {
    return (static_cast<std::int64_t>(row / 128) * (groups / 4) + group / 4) * 512 + (row % 32) * 16 +
           ((row % 128) / 32) * 4 + group % 4;
}

// Flat CTA-local row -> weight row: 4 consecutive flats share one swizzle quartet (rows q, q+32, q+64, q+96).
__device__ __forceinline__ int weight_row(int flat) {
    return (flat / 128) * 128 + ((flat & 127) >> 2) + (flat & 3) * 32;
}

template <int K, int T, int kRowsPerWarp, int kBlockRows>
__global__ __launch_bounds__(kBlockRows / kRowsPerWarp * 32) void fp4_gemv_kernel(
    const std::uint8_t* __restrict__ xq, const std::uint8_t* __restrict__ xs, const std::uint8_t* __restrict__ wq,
    const std::uint8_t* __restrict__ ws, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out, int n) {
    constexpr int kGroups = K / 16, kPhases = kGroups / 32, kThreads = kBlockRows / kRowsPerWarp * 32;
    static_assert(kGroups % 32 == 0, "K must be a multiple of 512");
    __shared__ alignas(16) std::uint8_t w_scale[kBlockRows][kGroups];
    __shared__ alignas(16) std::uint8_t x_scale[T][kGroups];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int flat0 = blockIdx.x * kBlockRows;
    // Weight scales: per 64-column tile, the CTA's quartets are kBlockRows*4 contiguous bytes (one uint4 each).
    {
        constexpr int kQuartets = kBlockRows / 4;
        const int tile = flat0 / 128, q0 = (flat0 & 127) >> 2;
        for (int task = threadIdx.x; task < (kGroups / 4) * kQuartets; task += kThreads) {
            const int kt = task / kQuartets, quartet = task % kQuartets;
            const uint4 v = *reinterpret_cast<const uint4*>(
                ws + (static_cast<std::int64_t>(tile) * (kGroups / 4) + kt) * 512 + (q0 + quartet) * 16);
            const std::uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int quartile = 0; quartile < 4; ++quartile)  // 4 groups of row (q0+quartet) + 32*quartile
                *reinterpret_cast<std::uint32_t*>(&w_scale[quartet * 4 + quartile][kt * 4]) = words[quartile];
        }
        for (int task = threadIdx.x; task < T * kGroups; task += kThreads) {
            const int t = task / kGroups, g = task % kGroups;
            x_scale[t][g] = xs[sf_offset(t, g, kGroups)];
        }
        __syncthreads();
    }

    const int local0 = warp * kRowsPerWarp;
    int row[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) row[r] = weight_row(flat0 + local0 + r);

    float acc[kRowsPerWarp][T] = {};
#pragma unroll kUnroll
    for (int phase = 0; phase < kPhases; ++phase) {
        const int g = phase * 32 + lane;
        std::uint32_t wv[kRowsPerWarp][8];
        float wsc[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            decode16(*reinterpret_cast<const uint2*>(wq + static_cast<std::int64_t>(row[r]) * (K / 2) + g * 8),
                     wv[r]);
            wsc[r] = e4m3(w_scale[local0 + r][g]);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            std::uint32_t xv[8];
            decode16(*reinterpret_cast<const uint2*>(xq + static_cast<std::int64_t>(t) * (K / 2) + g * 8), xv);
            const float xsc = e4m3(x_scale[t][g]);
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                std::uint32_t h = 0;  // f16x2 {even, odd} partial sums, exact
#pragma unroll
                for (int i = 0; i < 8; ++i) h = hfma2(wv[r][i], xv[i], h);
                acc[r][t] = fmaf(group_sum(h), wsc[r] * xsc, acc[r][t]);
            }
        }
    }
    const float a = *alpha;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) out[static_cast<std::int64_t>(t) * n + row[r]] = __float2bfloat16_rn(v * a);
        }
}

template <int K, int R, int BR>
void launch(int m, const std::uint8_t* xq, const std::uint8_t* xs, const std::uint8_t* wq, const std::uint8_t* ws,
            const float* alpha, __nv_bfloat16* out, int n, cudaStream_t stream) {
    const dim3 grid(n / BR), block(BR / R * 32);
#define FP4_GEMV_CASE(TT) \
    case TT: fp4_gemv_kernel<K, TT, R, BR><<<grid, block, 0, stream>>>(xq, xs, wq, ws, alpha, out, n); break;
    switch (m) {
        FP4_GEMV_CASE(1) FP4_GEMV_CASE(2) FP4_GEMV_CASE(3) FP4_GEMV_CASE(4)
        FP4_GEMV_CASE(5) FP4_GEMV_CASE(6) FP4_GEMV_CASE(7) FP4_GEMV_CASE(8)
        default: TORCH_CHECK(false, "fp4_gemv: M must be 1..8, got ", m);
    }
#undef FP4_GEMV_CASE
}
}  // namespace

// xq [M, K/2] uint8 (packed e2m1), xs swizzled e4m3 [>= pad128(M), K/16], wq [N, K/2], ws swizzled [N, K/16],
// alpha float32 [1] (device). Returns bf16 [M, N].
// rows_per_warp 1 or 2, block_rows 8 or 16 (0 = tuned defaults). Use for M <= 5 (down) / M <= 4 (gate_up);
// above that b12x is as fast or faster.
torch::Tensor fp4_gemv(torch::Tensor xq, torch::Tensor xs, torch::Tensor wq, torch::Tensor ws, torch::Tensor alpha,
                       int64_t rows_per_warp, int64_t block_rows) {
    const int m = xq.size(0), k = xq.size(1) * 2, n = wq.size(0);
    TORCH_CHECK(wq.size(1) * 2 == k && n % 128 == 0, "fp4_gemv: shape mismatch (N % 128 == 0 required)");
    TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous() && xs.is_contiguous() && ws.is_contiguous(),
                "fp4_gemv: contiguous operands required");
    TORCH_CHECK(xs.numel() >= static_cast<int64_t>((m + 127) / 128 * 128) * (k / 16) && ws.numel() == static_cast<int64_t>(n) * (k / 16),
                "fp4_gemv: scale sizes");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 && alpha.numel() == 1, "fp4_gemv: alpha float32[1]");
    const at::cuda::CUDAGuard guard(xq.device());
    auto out = torch::empty({m, n}, xq.options().dtype(torch::kBFloat16));
    auto p = [](const torch::Tensor& t) { return static_cast<const std::uint8_t*>(t.data_ptr()); };
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    // Defaults from the PRO 6000 sweep: 1 row/warp for the K=17408 down projection, 2 otherwise; 16 rows/CTA.
    const int r = rows_per_warp ? static_cast<int>(rows_per_warp) : (k == 17408 ? 1 : 2);
    const int br = block_rows ? static_cast<int>(block_rows) : 16;
#define FP4_GEMV_ARGS m, p(xq), p(xs), p(wq), p(ws), alpha.data_ptr<float>(), o, n, stream
#define FP4_GEMV_K(KK)                                                                    \
    case KK:                                                                              \
        if (br == 16 && r == 1) launch<KK, 1, 16>(FP4_GEMV_ARGS);                         \
        else if (br == 16 && r == 2) launch<KK, 2, 16>(FP4_GEMV_ARGS);                    \
        else if (br == 8 && r == 1) launch<KK, 1, 8>(FP4_GEMV_ARGS);                      \
        else if (br == 8 && r == 2) launch<KK, 2, 8>(FP4_GEMV_ARGS);                      \
        else TORCH_CHECK(false, "fp4_gemv: (block_rows, rows_per_warp) in {8,16} x {1,2}"); \
        break;
    switch (k) {
        FP4_GEMV_K(17408)
        FP4_GEMV_K(5120)
        default: TORCH_CHECK(false, "fp4_gemv: unsupported K ", k);
    }
#undef FP4_GEMV_K
#undef FP4_GEMV_ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) { mod.def("fp4_gemv", &fp4_gemv, pybind11::arg("xq"), pybind11::arg("xs"), pybind11::arg("wq"), pybind11::arg("ws"), pybind11::arg("alpha"), pybind11::arg("rows_per_warp") = 0, pybind11::arg("block_rows") = 0); }
