// GDN MTP decode with deferred state stores (hetero-qwen27b). Derived from vLLM v0.30.0
// csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (Apache-2.0, copy kept next to this file).
//
// Stock: every spec-decode step stores the fp32 recurrent state after EVERY token (k+1 slots x 3 MiB per sequence per
// layer for Qwen3.8-27B), only to read back slot[accepted-1] next step. Profile: the stores land as DRAM write
// traffic on the following kernels (down GEMM 37 -> 52 us at c=4).
// Here:
//   gdn_mtp_defer   (kernel A, per layer, in the forward): same outputs as stock (same code), but instead of the k+1
//                   state stores it saves each token's normalised k (fp32), v (bf16), decay and beta, plus the source /
//                   destination slots and an epoch stamp.
//   gdn_materialize (kernel B, once per step for all layers, after the accepted counts are known and before vLLM's
//                   mamba postprocess): reloads the source state, replays the accepted tokens with the same per-token
//                   update code, and stores slot[accepted-1] (and slot[bias] when the align-mode postprocess will copy
//                   a block-boundary state). Those are the only slots anything reads afterwards.
// kDefer=false compiles the stock kernel unchanged (reference for the unit test).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

__device__ __forceinline__ void cp_async_16b(float* smem_ptr, const float* gmem_ptr) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

__device__ __forceinline__ void copy_state_chunk(float* shared_state, const float* state, int chunk, int thread,
                                                 int threads) {
  constexpr int kElementsPerCopy = 4;
  constexpr int kCopiesPerChunk = kChunkV * kDimK / kElementsPerCopy;
  const int stage = chunk % 2;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * kChunkV * kDimK + element, state + chunk * kChunkV * kDimK + element);
  }
  cp_async_commit();
}

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

// Per-layer deferred-store buffers (layer offset applied by the host). S = slots per row (k+1).
struct DeferBufs {
  float* kbuf;          // [R][H][S][128]   normalised k
  __nv_bfloat16* vbuf;  // [R][HV][S][128]  v
  float* gbuf;          // [R][HV][S][2]    decay, beta
  int* meta;            // [R][3 + S]       epoch stamp, source slot, tokens, destination slots
  const int* epoch;
  int S;
};

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// One token of the delta rule on this thread's 4 rows x 4 k-elements, verbatim from the stock token loop. Both kernels
// call it, so the state arithmetic (incl. FMA contraction and the shuffle order) is the same instruction sequence.
template <bool kOut>
__device__ __forceinline__ void token_update(float (&h)[kRowsPerWarp][4], const float (&k_values)[4],
                                             const float (&q_values)[4], const __nv_bfloat16* v_row, int chunk,
                                             const int (&rows)[kRowsPerWarp], float decay, float beta,
                                             float (&dot_hq_out)[kRowsPerWarp]) {
  float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] *= decay;
      dot_hk[row] += h[row][i] * k_values[i];
    }
  }
  const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
  const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
  const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y, dot_hk_23.x, dot_hk_23.y};

  float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const int value = chunk * kChunkV + rows[row];
    const float delta = (__bfloat162float(v_row[value]) - reduced_hk[row]) * beta;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] += k_values[i] * delta;
      if constexpr (kOut) dot_hq[row] += h[row][i] * q_values[i];
    }
  }
  if constexpr (kOut) {
    const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
    const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
    dot_hq_out[0] = dot_hq_01.x;
    dot_hq_out[1] = dot_hq_01.y;
    dot_hq_out[2] = dot_hq_23.x;
    dot_hq_out[3] = dot_hq_23.y;
  }
}

__device__ __forceinline__ void store_rows(float* destination_state, int chunk, const int (&rows)[kRowsPerWarp],
                                           int k_base, const float (&h)[kRowsPerWarp][4]) {
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const int value = chunk * kChunkV + rows[row];
    *reinterpret_cast<float4*>(destination_state + value * kDimK + k_base) =
        make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
  }
}

template <int ValueHeadsPerKeyHead, bool SigmoidGate, bool kDefer>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_post_conv_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, float* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps, GdnDecodeStrides strides, DeferBufs defer) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot = accepted > 0 && accepted <= state_indices_width
                              ? state_indices[request * state_indices_width + accepted - 1]
                              : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  if constexpr (kDefer) {
    if (value_head == 0 && tid < 3 + state_indices_width) {
      int* meta = defer.meta + request * (3 + defer.S);
      meta[tid] = tid == 0   ? *defer.epoch
                  : tid == 1 ? source_slot
                  : tid == 2 ? num_tokens
                             : state_indices[request * state_indices_width + tid - 3];
    }
  }
  __shared__ float shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  float* source_state = state + static_cast<int64_t>(source_slot) * strides.state_slot + value_head * kDimV * kDimK;
  copy_state_chunk(&shared_state[0][0][0], source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
      if constexpr (kDefer) {
        if (value_head % ValueHeadsPerKeyHead == 0) {
          defer.kbuf[((static_cast<int64_t>(request) * H + key_head) * defer.S + t) * kDimK + dim] = shared_k[t][dim];
        }
        defer.vbuf[((static_cast<int64_t>(request) * HV + value_head) * defer.S + t) * kDimV + dim] = shared_v[t][dim];
      }
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) * softplus_fast(a_value + load_dt_bias(dt_bias, value_head, dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
      if constexpr (kDefer) {
        float* g2 = defer.gbuf + ((static_cast<int64_t>(request) * HV + value_head) * defer.S + t) * 2;
        g2[0] = shared_decay[t];
        g2[1] = shared_beta[t];
      }
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) copy_state_chunk(&shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value = *reinterpret_cast<const float4*>(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float dot_hq[kRowsPerWarp];
      token_update<true>(h, k_values, q_values, shared_v[t], chunk, rows, shared_decay[t], shared_beta[t], dot_hq);
      if (lane == 0) {
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row)
          shared_out[t][chunk * kChunkV + rows[row]] = __float2bfloat16(dot_hq[row]);
      }
      if constexpr (!kDefer) {
        const int destination_slot = state_indices[request * state_indices_width + t];
        if (destination_slot > 0)
          store_rows(state + static_cast<int64_t>(destination_slot) * strides.state_slot + value_head * kDimV * kDimK,
                     chunk, rows, k_base, h);
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input =
          __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

// Kernel B. grid (R, HV, L): one CTA per (spec row, value head, GDN layer).
__global__ __launch_bounds__(kThreads, 2) void gdn_materialize_kernel(
    const int64_t* __restrict__ state_ptrs, int64_t slot_stride, const float* __restrict__ kbuf, int64_t kbuf_ls,
    const __nv_bfloat16* __restrict__ vbuf, int64_t vbuf_ls, const float* __restrict__ gbuf, int64_t gbuf_ls,
    const int* __restrict__ meta_base, int64_t meta_ls, const int* __restrict__ epoch, int S,
    const int* __restrict__ rowmap, const int* __restrict__ num_accepted, const int* __restrict__ num_computed,
    const int* __restrict__ num_scheduled, const int* __restrict__ num_draft, int block_size, int H, int HV,
    int value_heads_per_key_head, const int* __restrict__ idx_map, int precomputed) {
  const int r = blockIdx.x;
  const int value_head = blockIdx.y;
  const int layer = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int* meta = meta_base + layer * meta_ls + r * (3 + S);
  if (meta[0] != *epoch) return;  // kernel A did not run for this row in this step
  const int source_slot = meta[1];
  const int num_tokens = meta[2];
  // V1 runner: per-row arrays indexed by batch row. V2 runner (idx_map != null): indexed by request-state slot.
  const int row_idx = idx_map != nullptr ? idx_map[rowmap[r]] : rowmap[r];
  const int accepted = num_accepted[row_idx];
  if (accepted < 1 || accepted > num_tokens) {
    __trap();  // cannot happen for a spec row (accepted <= k+1 = tokens); never write a wrong state silently
  }
  const int t_last = accepted - 1;
  int t_bias = -1;  // vLLM align postprocess: copies the state at accept_token_bias across a block boundary
  if (block_size > 0) {
    // precomputed (V2): num_computed already holds the post-step count = running + accepted - 1
    const int new_computed =
        precomputed ? num_computed[row_idx]
                    : num_computed[row_idx] + num_scheduled[row_idx] - num_draft[row_idx] + accepted - 1;
    const int running = new_computed - accepted + 1;
    const int aligned = (new_computed / block_size) * block_size;
    if (aligned >= running) t_bias = aligned - running;
  }
  const int dst_last = meta[3 + t_last];
  const int dst_bias = t_bias >= 0 ? meta[3 + t_bias] : 0;

  float* state = reinterpret_cast<float*>(state_ptrs[layer]);
  const float* source_state = state + static_cast<int64_t>(source_slot) * slot_stride + value_head * kDimV * kDimK;
  const int key_head = value_head / value_heads_per_key_head;
  const float* kb = kbuf + layer * kbuf_ls + (static_cast<int64_t>(r) * H + key_head) * S * kDimK;
  const __nv_bfloat16* vb = vbuf + layer * vbuf_ls + (static_cast<int64_t>(r) * HV + value_head) * S * kDimV;
  const float* gb = gbuf + layer * gbuf_ls + (static_cast<int64_t>(r) * HV + value_head) * S * 2;

  __shared__ float shared_state[2][kChunkV][kDimK];
  copy_state_chunk(&shared_state[0][0][0], source_state, 0, tid, kThreads);

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;
  const float q_dummy[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) copy_state_chunk(&shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value = *reinterpret_cast<const float4*>(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }
    for (int t = 0; t <= t_last; ++t) {
      const float4 k4 = *reinterpret_cast<const float4*>(kb + t * kDimK + k_base);
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float unused[kRowsPerWarp];
      token_update<false>(h, k_values, q_dummy, vb + t * kDimV, chunk, rows, gb[t * 2], gb[t * 2 + 1], unused);
      if (t == t_last && dst_last > 0)
        store_rows(state + static_cast<int64_t>(dst_last) * slot_stride + value_head * kDimV * kDimK, chunk, rows,
                   k_base, h);
      else if (t == t_bias && dst_bias > 0)
        store_rows(state + static_cast<int64_t>(dst_bias) * slot_stride + value_head * kDimV * kDimK, chunk, rows,
                   k_base, h);
    }
  }
}

template <int VPK, bool Sig, bool Defer>
void launch_a(const torch::Tensor& mixed_qkv, const torch::Tensor& a, const torch::Tensor& b,
              const torch::Tensor& a_log, const torch::Tensor& dt_bias, const torch::Tensor& state_indices,
              const torch::Tensor& cu_seqlens, const torch::Tensor& num_accepted, torch::Tensor& state,
              const torch::Tensor& output_gate, const torch::Tensor& norm_weight, torch::Tensor& out, int H, int HV,
              int dt_bias_type, double scale, double norm_eps, GdnDecodeStrides strides, DeferBufs defer) {
  const dim3 grid(state_indices.size(0), HV);
  gdn_decode_post_conv_mtp_kernel<VPK, Sig, Defer><<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      a_log.data_ptr<float>(), dt_bias.data_ptr(), state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
      num_accepted.data_ptr<int>(), state.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(output_gate.data_ptr()), norm_weight.data_ptr(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, static_cast<int>(state_indices.size(1)), dt_bias_type,
      norm_weight.scalar_type() == at::kBFloat16, static_cast<float>(scale), static_cast<float>(norm_eps), strides,
      defer);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// defer=false: the stock kernel (state stores). defer=true: kernel A (buffers required).
void gdn_mtp(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
             torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor state,
             torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out, double scale, double norm_eps,
             bool sigmoid_gate, bool defer, c10::optional<torch::Tensor> kbuf, c10::optional<torch::Tensor> vbuf,
             c10::optional<torch::Tensor> gbuf, c10::optional<torch::Tensor> meta,
             c10::optional<torch::Tensor> epoch) {
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 4 && state.size(2) == kDimV &&
              state.size(3) == kDimK && state.stride(1) == kDimV * kDimK && state.stride(2) == kDimK &&
              state.stride(3) == 1 && state.stride(0) % 4 == 0, "state must be fp32 [slots, HV, 128, 128]");
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kBFloat16 && mixed_qkv.dim() == 2 && mixed_qkv.stride(1) == 1);
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.is_contiguous() && state_indices.dim() == 2 &&
              state_indices.size(1) <= kMaxMtpTokens);
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.is_contiguous());
  TORCH_CHECK(num_accepted.scalar_type() == at::kInt && num_accepted.is_contiguous());
  TORCH_CHECK(output_gate.dim() == 3 && output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV);
  TORCH_CHECK(out.is_contiguous() && a.stride(1) == 1 && b.stride(1) == 1);
  const int HV = state.size(1);
  const int H = (mixed_qkv.size(1) - HV * kDimV) / (2 * kDimK);
  TORCH_CHECK(HV % H == 0, "HV % H");
  const int vpk = HV / H;
  const auto dts = dt_bias.scalar_type();
  const int dt_bias_type = dts == at::kFloat ? kDtBiasFloat32 : (dts == at::kBFloat16 ? kDtBiasBFloat16 : kDtBiasFloat16);
  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0), state.stride(0)};
  DeferBufs d{nullptr, nullptr, nullptr, nullptr, nullptr, 0};
  if (defer) {
    TORCH_CHECK(kbuf && vbuf && gbuf && meta && epoch, "defer needs buffers");
    d = DeferBufs{kbuf->data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(vbuf->data_ptr()), gbuf->data_ptr<float>(),
                  meta->data_ptr<int>(), epoch->data_ptr<int>(), static_cast<int>(meta->size(-1)) - 3};
    TORCH_CHECK(state_indices.size(1) <= d.S && state_indices.size(0) <= meta->size(0), "defer buffers too small");
  }
#define QG_LAUNCH(VPK)                                                                                          \
  if (sigmoid_gate) {                                                                                           \
    if (defer) launch_a<VPK, true, true>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens,           \
                                         num_accepted, state, output_gate, norm_weight, out, H, HV,            \
                                         dt_bias_type, scale, norm_eps, strides, d);                           \
    else launch_a<VPK, true, false>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted,  \
                                    state, output_gate, norm_weight, out, H, HV, dt_bias_type, scale, norm_eps,\
                                    strides, d);                                                               \
  } else {                                                                                                      \
    if (defer) launch_a<VPK, false, true>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens,          \
                                          num_accepted, state, output_gate, norm_weight, out, H, HV,           \
                                          dt_bias_type, scale, norm_eps, strides, d);                          \
    else launch_a<VPK, false, false>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted, \
                                     state, output_gate, norm_weight, out, H, HV, dt_bias_type, scale,         \
                                     norm_eps, strides, d);                                                    \
  }
  switch (vpk) {
    case 1: QG_LAUNCH(1) break;
    case 2: QG_LAUNCH(2) break;
    case 3: QG_LAUNCH(3) break;
    case 4: QG_LAUNCH(4) break;
    case 8: QG_LAUNCH(8) break;
    default: TORCH_CHECK(false, "HV/H must be 1, 2, 3, 4 or 8");
  }
#undef QG_LAUNCH
}

void gdn_materialize(torch::Tensor state_ptrs, int64_t slot_stride, torch::Tensor kbuf, torch::Tensor vbuf,
                     torch::Tensor gbuf, torch::Tensor meta, torch::Tensor epoch, torch::Tensor rowmap,
                     torch::Tensor num_accepted, c10::optional<torch::Tensor> num_computed,
                     c10::optional<torch::Tensor> num_scheduled, c10::optional<torch::Tensor> num_draft,
                     int64_t block_size, int64_t rows, int64_t H, int64_t HV, c10::optional<torch::Tensor> idx_map,
                     bool precomputed) {
  const int L = kbuf.size(0);
  TORCH_CHECK(state_ptrs.scalar_type() == at::kLong && state_ptrs.numel() == L);
  TORCH_CHECK(rows <= meta.size(1) && rows <= rowmap.numel());
  if (rows <= 0) return;
  const bool copy = block_size > 0;
  TORCH_CHECK(!copy || (num_computed && (precomputed || (num_scheduled && num_draft))));
  const dim3 grid(rows, HV, L);
  gdn_materialize_kernel<<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      state_ptrs.data_ptr<int64_t>(), slot_stride, kbuf.data_ptr<float>(), kbuf.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(vbuf.data_ptr()), vbuf.stride(0), gbuf.data_ptr<float>(), gbuf.stride(0),
      meta.data_ptr<int>(), meta.stride(0), epoch.data_ptr<int>(), static_cast<int>(meta.size(-1)) - 3,
      rowmap.data_ptr<int>(), num_accepted.data_ptr<int>(), copy ? num_computed->data_ptr<int>() : nullptr,
      copy && !precomputed ? num_scheduled->data_ptr<int>() : nullptr,
      copy && !precomputed ? num_draft->data_ptr<int>() : nullptr,
      static_cast<int>(block_size), static_cast<int>(H), static_cast<int>(HV), static_cast<int>(HV / H),
      idx_map ? idx_map->data_ptr<int>() : nullptr, precomputed ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_mtp", &gdn_mtp);
  m.def("gdn_materialize", &gdn_materialize);
}
