#include <type_traits>
// v13b: v13 + no key tiles (no page reads) when L < q tokens (graph padding rows; output 0). v13: v10h + FP8_NW macro (NW 4 build = spec-verify decode, inactive warps skip compute) + graph-safe split mode (max_seq_len <= 0: split ranges sized on the device) + cp.async row step NT / 16 (was a fixed 16 = NW 8 only); NW 8 output identical to v10h. v10h: v10g + barrier after the prologue QK (race fix). v10g: v10d + one S buffer (QK(t+1) after softmax(t): no S copy) + Q A fragments resident in registers (bit-identical). v10d: v9 + q scale folded into the exp FFMA and the P x 256 into the exponent (noise-level), unmasked tiles without the causal-mask code, cp.async offsets precomputed (bit-identical). v9: v8b + merge-aware split choice. v8b: v8 + select-free exp. PV as O^T = V^T P^T (V = A operand via one x2 trans ldmatrix, P = B operand). FP8 tensor-core paged extend attention for sm_120a (Qwen3.8-27B full-attention layers: HQ 24, HK 4, D 256),
//. Numerics = vLLM hook QWEN38_FP8ATTN_EMU (mode qp): Q -> e4m3 with a per (token, head) scale
// (amax / 448), K / V = the FP8 cache bytes, S = Q8.K8 in fp32 (mma.sync m16n8k32 kind::f8f6f4), online softmax in
// the log2 domain, P * 256 -> e4m3, O += P8.V8 (fp32), row sum from the fp32 P.
// Layout: q [nq, HQ, D] bf16; k / v cache [pages, PAGE=64, HK, D] e4m3 with arbitrary page / token / head strides
// (bytes); page_table [width] int32 (one request); out [nq, HQ, D] bf16.
// CTA = one KV head x 128 packed rows (row = token * 6 + head-in-group, ~21 tokens) x one KV split; 8 warps x 16
// rows (2 warps per SM sub-partition: 6 warps left two tensor pipes half idle, 777 vs 1008 TF in an MMA-rate microbenchmark).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
#ifndef FP8_NW
#define FP8_NW 8                                // warps (x 16 rows) per CTA; 4 for spec-verify decode (8 tokens x 6 = 48 rows)
#endif
constexpr int D = 256, GRP = 6, HK = 4, HQ = 24, PAGE = 64, TQ = 16, NW = FP8_NW, NT = NW * 32, ROWS = NW * TQ;
static_assert(NW % 4 == 0, "cp.async chunks: NT / 16 rows per step must keep row & 7 (swizzle)");
constexpr int KT = 64;                       // keys per tile (= one page)
constexpr int NST = 2;                       // cp.async pipeline stages

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// two f32 -> e4m3x2 (lo byte = x0, hi byte = x1), round to nearest, saturate
__device__ __forceinline__ uint16_t f2_to_e4m3x2(float x0, float x1) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(x1), "f"(x0));
  return r;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// K tile [64 keys][256 B]: 16-byte chunk c of row r stored at chunk c ^ (r & 7)
__device__ __forceinline__ int k_off(int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); }
// Vt tile [256 dims][64 B]: chunk c (0..3) of row d stored at chunk c ^ ((d >> 1) & 3)
__device__ __forceinline__ int vt_off(int d, int c) { return d * 64 + ((c ^ ((d >> 1) & 3)) << 4); }
// Q tile per warp [16 rows][256 B], same swizzle as K
__device__ __forceinline__ int q_off(int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); }

struct Params {
  const uint8_t* q8;      // [ntok, HQ, D] e4m3
  const float* qsc;       // [ntok, HQ] fp32 = q amax/448 * k_descale * sm_scale * log2(e)
  const uint8_t* kc;
  const uint8_t* vc;
  long long k_ps, k_ts, k_hs, v_ps, v_ts, v_hs;   // byte strides: page, token, head
  const int* pt;          // [nreq, pt_stride] 64-token page ids
  int pt_stride;
  const int* seq_lens;    // [nreq] total length (cached + new)
  const int* cu_q;        // [nreq + 1] query row offsets
  int nreq, ntok, nsplit, tiles_per_split;
  float v_out_scale;      // v_descale (P and l both carry the 256)
  float* opart;           // [nsplit, ntok, HQ, D]   (nsplit > 1)
  float* mpart;           // [nsplit, ntok, HQ]  (log2 domain)
  float* lpart;           // [nsplit, ntok, HQ]
  __nv_bfloat16* out;     // [ntok, HQ, D]           (nsplit == 1: written directly)
};

__device__ __forceinline__ void ldsm_16x16x2_trans_b8(uint32_t addr, uint32_t (&r)[4]) {
  // lanes 0-15: rows 0-15 of matrix 0, lanes 16-31: rows of matrix 1; r0 r1 = matrix 0, r2 r3 = matrix 1
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_16x16_trans_b8(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  // lanes 0-15 give rows 0-15 (16 B each); lane t gets r0 = rows 4(t%4)..+3 at byte col t/4, r1 = same rows at col t/4+8
  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__global__ void __launch_bounds__(NT, 1) fp8_extend_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* qs = smem;                          // NW * 16 * 256 = 24 KB
  uint8_t* ks = qs + NW * TQ * 256;            // NST x 16 KB
  uint8_t* vs = ks + NST * KT * 256;           // NST x 16 KB, natural [key][256 B], K swizzle
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.y, split = blockIdx.z;
  // row block -> (request, first packed row): blocks are laid out request by request, ceil(nq * 6 / 128) each
  __shared__ int s_req, s_rb;
  if (tid == 0) {
    int rbg = blockIdx.x, acc = 0, req = -1, rbl = 0;
    for (int q = 0; q < p.nreq; ++q) {
      int nb = ((p.cu_q[q + 1] - p.cu_q[q]) * GRP + ROWS - 1) / ROWS;
      if (rbg < acc + nb) { req = q; rbl = rbg - acc; break; }
      acc += nb;
    }
    s_req = req; s_rb = rbl;
  }
  __syncthreads();
  const int req = s_req;
  if (req < 0) return;
  const int qbase = p.cu_q[req], nq_r = p.cu_q[req + 1] - qbase, L = p.seq_lens[req];
  const int* ptr = p.pt + (long long)req * p.pt_stride;
  const int R0 = s_rb * ROWS, nrows = nq_r * GRP;  // packed rows of this KV head within the request
  const int qpos0 = L - nq_r + R0 / GRP;          // smallest query position in the CTA
  const int last_pos = min(L - nq_r + (min(R0 + ROWS, nrows) - 1) / GRP, L - 1);
  const int ntiles = last_pos < 0 ? 0 : last_pos / KT + 1;   // L < q tokens (graph padding rows): no keys, no page reads
  // tiles_per_split 0 (graph-safe mode): each CTA splits its own range, >= 4 tiles per split, trailing splits empty
  const int tps_ = p.tiles_per_split > 0 ? p.tiles_per_split : max(4, (ntiles + p.nsplit - 1) / p.nsplit);
  const int tbeg = split * tps_, tend = min(ntiles, tbeg + tps_);
  const int r = lane >> 2, qd = lane & 3;

  // Q tile -> smem (warp w: packed rows R0 + 16w .. +15; rows past the end are zero)
  for (int i = lane; i < TQ * 16; i += 32) {
    int row = i >> 4, c = i & 15, pr = R0 + warp * TQ + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (pr < nrows) v = *reinterpret_cast<const uint4*>(p.q8 + ((long long)(qbase + pr / GRP) * HQ + g * GRP + pr % GRP) * D + c * 16);
    *reinterpret_cast<uint4*>(qs + warp * TQ * 256 + q_off(row, c)) = v;
  }
  float rs[2];
  int tok[2], hd[2], qpos[2];
#pragma unroll
  for (int h2 = 0; h2 < 2; ++h2) {
    int pr = R0 + warp * TQ + r + 8 * h2;
    tok[h2] = pr / GRP; hd[h2] = g * GRP + pr % GRP;
    qpos[h2] = L - nq_r + tok[h2];
    rs[h2] = pr < nrows ? p.qsc[(long long)(qbase + tok[h2]) * HQ + hd[h2]] : 1.f;   // > 0 always (q rows past the end are 0)
    tok[h2] = pr < nrows ? qbase + tok[h2] : -1;   // global row; -1 = inactive
  }
  // O^T accumulators: ot[dg][h] = m16 (dims 16 dg + r, + 8) x n8 (rows 8 h + 2 qd, + 1)
  float ot[16][2][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) ot[i][h][0] = ot[i][h][1] = ot[i][h][2] = ot[i][h][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  // Software pipeline (FA3-style, within each warp): QK of tile t+1 is issued next to the softmax of tile t, then
  // PV of tile t. Copy groups: G(t) = {K(t+2), V(t+1)} issued at the top of iteration t; K / V double-buffered by tile.
  const uint8_t* qb = qs + warp * TQ * 256;
  uint32_t qa[8][4];   // Q A fragments, resident for the whole KV loop (k-step kk: dims 32 kk .. +31)
  // cp.async chunks of a 64 x 256 B tile: thread tid copies rows tid / 16 + (NT / 16) j, 16 B column tid % 16.
  // The swizzle depends on row & 7 only, so chunk j sits NT / 16 rows per step after chunk 0 (NT / 16 is a multiple of 8): offsets computed once.
  const uint32_t cp_soff = k_off(tid >> 4, tid & 15);
  const long long cp_koff = (long long)(tid >> 4) * p.k_ts + (tid & 15) * 16, cp_kstep = (long long)(NT / 16) * p.k_ts;
  const long long cp_voff = (long long)(tid >> 4) * p.v_ts + (tid & 15) * 16, cp_vstep = (long long)(NT / 16) * p.v_ts;
  auto issue_k = [&](int tile) {
    const uint8_t* base = p.kc + (long long)ptr[tile] * p.k_ps + (long long)g * p.k_hs + cp_koff;
    const uint32_t dst = smem_u32(ks + (tile & 1) * KT * 256) + cp_soff;
#pragma unroll
    for (int j = 0; j < KT * 16 / NT; ++j) cp_async16(dst + j * (NT / 16) * 256, base + j * cp_kstep);
  };
  auto issue_v = [&](int tile) {
    const uint8_t* base = p.vc + (long long)ptr[tile] * p.v_ps + (long long)g * p.v_hs + cp_voff;
    const uint32_t dst = smem_u32(vs + (tile & 1) * KT * 256) + cp_soff;
#pragma unroll
    for (int j = 0; j < KT * 16 / NT; ++j) cp_async16(dst + j * (NT / 16) * 256, base + j * cp_vstep);
  };
  // S = Q8 K8^T for one 64-key tile; S n8-tile i, column c holds key 32(i/4) + 16((i%4)/2) + 4(c/2) + 2(i%2) + c%2
  auto qk = [&](int tile, float (&s)[8][4]) {
    const uint8_t* kb = ks + (tile & 1) * KT * 256;
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; kk += 2) {
      const uint32_t (&a)[2][4] = *reinterpret_cast<const uint32_t (*)[2][4]>(&qa[kk]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int mi = lane >> 3, cl = lane & 7, il = i & 3;
        int row = 32 * (i >> 2) + 16 * (il >> 1) + 4 * (cl >> 1) + 2 * (il & 1) + (cl & 1);
        int c = 2 * (kk + (mi >> 1)) + (mi & 1);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(smem_u32(kb + k_off(row, c)), b0, b1, b2, b3);
        mma_fp8(s[i], a[0][0], a[0][1], a[0][2], a[0][3], b0, b1);
        mma_fp8(s[i], a[1][0], a[1][1], a[1][2], a[1][3], b2, b3);
      }
    }
  };
  // online softmax of S (log2 domain) -> P8 A fragments; rescales O when a row max moved
  // MT: std::true_type for the tiles that reach past the smallest query position (causal mask), else false_type;
  // the unmasked instance carries no per-element key compare / select
  auto softmax = [&](auto MT, int tile, float (&s)[8][4], uint32_t (&pb)[2][2][2]) {
    const int key0 = tile * KT;
    // rs > 0, so max(s * rs) = rs * max(s) exactly (rounding is monotone): the row max is taken on the raw scores
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int h2 = e >> 1;
        float v = s[i][e];
        if constexpr (decltype(MT)::value) {
          int key = key0 + 32 * (i >> 2) + 16 * ((i & 3) >> 1) + 4 * qd + 2 * (i & 1) + (e & 1);
          if (key > qpos[h2]) v = -INFINITY;
        }
        s[i][e] = v;
        mx[h2] = fmaxf(mx[h2], v);
      }
    float alpha[2];
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xffffffff, mx[h2], 1));
      mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xffffffff, mx[h2], 2));
      mx[h2] = fmaxf(m[h2], mx[h2] * rs[h2]);
      alpha[h2] = mx[h2] == -INFINITY ? 1.f : ex2(m[h2] - mx[h2]);
      m[h2] = mx[h2];
    }
    float ls[2] = {0.f, 0.f};
    // m = -inf only if every s of the row is -inf, and then ex2(-inf) = 0: same values without a per-element select.
    // P is scaled by 256 (e4m3 range) inside the exponent: 256 * 2^x = 2^(x + 8); l carries the same factor.
    const float mb[2] = {(m[0] == -INFINITY ? 0.f : m[0]) - 8.f, (m[1] == -INFINITY ? 0.f : m[1]) - 8.f};
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int h2 = e >> 1;
        float pv = ex2(fmaf(s[i][e], rs[h2], -mb[h2]));
        s[i][e] = pv;
        ls[h2] += pv;
      }
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) l[h2] = l[h2] * alpha[h2] + ls[h2];
    if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) {
      // alpha of the O^T rows this lane holds: row 2qd (+1) lives in lanes 4 * (2qd (+1)), as h2 = 0 / 1 for rows + 8
      float a00 = __shfl_sync(0xffffffff, alpha[0], 8 * qd), a01 = __shfl_sync(0xffffffff, alpha[0], 8 * qd + 4);
      float a10 = __shfl_sync(0xffffffff, alpha[1], 8 * qd), a11 = __shfl_sync(0xffffffff, alpha[1], 8 * qd + 4);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        ot[i][0][0] *= a00; ot[i][0][1] *= a01; ot[i][0][2] *= a00; ot[i][0][3] *= a01;
        ot[i][1][0] *= a10; ot[i][1][1] *= a11; ot[i][1][2] *= a10; ot[i][1][3] *= a11;
      }
    }
    // P as the B operand of O^T += V^T P^T: pb[c][h] = {b0, b1} for key chunk c, rows 8h + r (same bytes as the
    // A fragments of the O = P V form, regrouped into register pairs)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float* s0 = s[4 * c]; const float* s1 = s[4 * c + 1];
      const float* s2 = s[4 * c + 2]; const float* s3 = s[4 * c + 3];
      pb[c][0][0] = (uint32_t)f2_to_e4m3x2(s0[0], s0[1]) | ((uint32_t)f2_to_e4m3x2(s1[0], s1[1]) << 16);
      pb[c][1][0] = (uint32_t)f2_to_e4m3x2(s0[2], s0[3]) | ((uint32_t)f2_to_e4m3x2(s1[2], s1[3]) << 16);
      pb[c][0][1] = (uint32_t)f2_to_e4m3x2(s2[0], s2[1]) | ((uint32_t)f2_to_e4m3x2(s3[0], s3[1]) << 16);
      pb[c][1][1] = (uint32_t)f2_to_e4m3x2(s2[2], s2[3]) | ((uint32_t)f2_to_e4m3x2(s3[2], s3[3]) << 16);
    }
  };
  auto pv = [&](int tile, const uint32_t (&pb)[2][2][2]) {
    const uint8_t* vb = vs + (tile & 1) * KT * 256;
#pragma unroll
    for (int dg = 0; dg < 16; ++dg) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        uint32_t a[4];                          // V^T fragment: dims 16 dg .. +15 x keys 32 c .. +31
        ldsm_16x16x2_trans_b8(smem_u32(vb + k_off(32 * c + lane, dg)), a);
        mma_fp8(ot[dg][0], a[0], a[1], a[2], a[3], pb[c][0][0], pb[c][0][1]);
        mma_fp8(ot[dg][1], a[0], a[1], a[2], a[3], pb[c][1][0], pb[c][1][1]);
      }
    }
  };

  __syncthreads();                              // Q tile in smem
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    int mi = lane >> 3, row = (lane & 7) + 8 * (mi & 1), c = 2 * kk + (mi >> 1);
    ldsm_x4(smem_u32(qb + q_off(row, c)), qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3]);
  }
  if (tbeg < tend) {
    issue_k(tbeg); issue_v(tbeg); cp_commit();
    if (tbeg + 1 < tend) issue_k(tbeg + 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    // One S buffer: softmax(t) packs S(t) into P8, then QK(t + 1) overwrites S, then PV(t) (no S copy; the other warp
    // of the scheduler covers the ALU / tensor overlap)
    float sc[8][4];
    // only the small-CTA build (spec-verify decode, NW 4: 48 of 64 rows) has whole warps without rows; NW 8 keeps
    // the branch-free loop
    const bool warp_active = NW >= 8 || R0 + warp * TQ < nrows;
    if (warp_active) qk(tbeg, sc);
    __syncthreads();                // every warp is done with K(tbeg) before iteration tbeg refills its buffer
    for (int t = tbeg; t < tend; ++t) {
      if (t + 2 < tend) issue_k(t + 2);          // into K(t)'s buffer: QK(t) ran in the previous iteration
      if (t + 1 < tend) issue_v(t + 1);
      cp_commit();
      cp_wait<1>();                 // K(t+1), V(t) landed
      __syncthreads();
      if (warp_active) {                            // a warp without valid rows only helps with the copies
        uint32_t pb[2][2][2];
        if (t * KT + KT - 1 > qpos0) softmax(std::true_type{}, t, sc, pb);
        else softmax(std::false_type{}, t, sc, pb);
        if (t + 1 < tend) qk(t + 1, sc);
        pv(t, pb);
      }
      __syncthreads();
    }
  }
  // ---- write the split's partial state ----
#pragma unroll
  for (int h2 = 0; h2 < 2; ++h2) {
    l[h2] += __shfl_xor_sync(0xffffffff, l[h2], 1);
    l[h2] += __shfl_xor_sync(0xffffffff, l[h2], 2);
  }
  // per-row stats by the row owners (lanes 4r: rows r, r + 8), as before
  if (p.nsplit > 1 && qd == 0) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      if (tok[h2] < 0) continue;
      long long row = ((long long)split * p.ntok + tok[h2]) * HQ + hd[h2];
      p.mpart[row] = m[h2];
      p.lpart[row] = l[h2];
    }
  }
  // O^T rows held here: 8h + 2qd + e (e = 0, 1); their l from lane 4 * (2qd + e), h2 = h
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float lr = __shfl_sync(0xffffffff, l[h], 4 * (2 * qd + e));
      int pr = R0 + warp * TQ + 8 * h + 2 * qd + e;
      if (pr >= nrows) continue;
      long long gtok = qbase + pr / GRP; int ghd = g * GRP + pr % GRP;
      if (p.nsplit == 1) {
        float inv = lr > 0.f ? p.v_out_scale / lr : 0.f;
        __nv_bfloat16* dst = p.out + (gtok * HQ + ghd) * D;
#pragma unroll
        for (int dg = 0; dg < 16; ++dg) {
          dst[16 * dg + r] = __float2bfloat16_rn(ot[dg][h][e] * inv);
          dst[16 * dg + r + 8] = __float2bfloat16_rn(ot[dg][h][2 + e] * inv);
        }
      } else {
        float* dst = p.opart + (((long long)split * p.ntok + gtok) * HQ + ghd) * D;
#pragma unroll
        for (int dg = 0; dg < 16; ++dg) {
          dst[16 * dg + r] = ot[dg][h][e] * p.v_out_scale;
          dst[16 * dg + r + 8] = ot[dg][h][2 + e] * p.v_out_scale;
        }
      }
    }
}

// one warp per (token, head): merge the splits, normalize, bf16
__global__ void merge_kernel(const float* opart, const float* mpart, const float* lpart, __nv_bfloat16* out, int nq,
                             int nsplit) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= nq * HQ) return;
  float M = -INFINITY;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, mpart[(long long)s * nq * HQ + w]);
  float acc[8] = {}, Ls = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    long long row = (long long)s * nq * HQ + w;
    float ms = mpart[row];
    if (ms == -INFINITY) continue;
    float f = ex2(ms - M);
    Ls += lpart[row] * f;
    const float4* src = reinterpret_cast<const float4*>(opart + row * D + lane * 8);
    float4 x0 = src[0], x1 = src[1];
    acc[0] += x0.x * f; acc[1] += x0.y * f; acc[2] += x0.z * f; acc[3] += x0.w * f;
    acc[4] += x1.x * f; acc[5] += x1.y * f; acc[6] += x1.z * f; acc[7] += x1.w * f;
  }
  float inv = Ls > 0.f ? 1.f / Ls : 0.f;
  __nv_bfloat16* dst = out + (long long)w * D + lane * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) dst[j] = __float2bfloat16(acc[j] * inv);
}

// one warp per (token, head): bf16 row -> e4m3 with scale amax/448
__global__ void quant_q_kernel(const __nv_bfloat16* q, uint8_t* q8, float* qsc, int rows, float post_scale) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= rows) return;
  const __nv_bfloat16* src = q + (long long)w * D + lane * 8;
  float x[8], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) { x[j] = __bfloat162float(src[j]); amax = fmaxf(amax, fabsf(x[j])); }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  float sc = fmaxf(amax / 448.f, 1e-30f), inv = 1.f / sc;
  uint32_t w0 = (uint32_t)f2_to_e4m3x2(x[0] * inv, x[1] * inv) | ((uint32_t)f2_to_e4m3x2(x[2] * inv, x[3] * inv) << 16);
  uint32_t w1 = (uint32_t)f2_to_e4m3x2(x[4] * inv, x[5] * inv) | ((uint32_t)f2_to_e4m3x2(x[6] * inv, x[7] * inv) << 16);
  *reinterpret_cast<uint2*>(q8 + (long long)w * D + lane * 8) = make_uint2(w0, w1);
  if (lane == 0) qsc[w] = sc * post_scale;
}
}  // namespace

// Batched extend: q [ntok, HQ, D] bf16 (requests back to back, cu_q device offsets); k_cache / v_cache e4m3 4-D
// [pages, PAGE, HK, D] views (any strides); page_table [nreq, width] int32 of 64-token pages; seq_lens [nreq] device;
// max_seq_len (host) bounds the split ranges. Writes out [ntok, HQ, D] bf16. No host <-> device syncs.
// max_seq_len <= 0 = graph-safe mode (CUDA-graph decode: lengths change between replays): the grid depends on
// ntok / nreq only (nsplit_req, or enough splits for ~one wave) and every CTA sizes its split range on the device.
void fp8_extend_batch(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                      torch::Tensor seq_lens, torch::Tensor cu_q, int64_t nreq, int64_t max_seq_len, double k_descale,
                      double v_descale, double sm_scale, torch::Tensor out, int64_t nsplit_req) {
  TORCH_CHECK(q.dim() == 3 && q.size(1) == HQ && q.size(2) == D && q.scalar_type() == at::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) >= q.size(0));
  TORCH_CHECK(k_cache.size(1) == PAGE && k_cache.size(2) == HK && k_cache.size(3) == D && k_cache.stride(3) == 1);
  TORCH_CHECK(v_cache.size(1) == PAGE && v_cache.size(2) == HK && v_cache.size(3) == D && v_cache.stride(3) == 1);
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.stride(1) == 1);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && cu_q.scalar_type() == at::kInt);
  int ntok = q.size(0);
  if (ntok == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  auto opts_f = q.options().dtype(at::kFloat);
  auto q8 = torch::empty({ntok, HQ, D}, q.options().dtype(at::kByte));
  auto qsc = torch::empty({ntok, HQ}, opts_f);
  const float LOG2E = 1.4426950408889634f;
  int rows = ntok * HQ;
  quant_q_kernel<<<(rows * 32 + 255) / 256, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                                                          q8.data_ptr<uint8_t>(), qsc.data_ptr<float>(), rows,
                                                          (float)(k_descale * sm_scale) * LOG2E);
  int nrb = (ntok * GRP + ROWS - 1) / ROWS + (int)nreq;   // upper bound on row blocks
  int max_tiles = (int)((std::max<int64_t>(max_seq_len, 1) - 1) / KT + 1);
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int nsplit = (int)nsplit_req;
  const bool graph_safe = max_seq_len <= 0;
  if (graph_safe && nsplit <= 0) {
    // one CTA per SM: the most splits whose CTAs still fit one wave (a few CTAs over = a nearly empty second wave).
    // Row blocks exact for uniform q lengths (spec-verify decode), else the upper bound.
    int rb = ntok % nreq == 0 ? (int)nreq * ((ntok / (int)nreq * GRP + ROWS - 1) / ROWS)
                              : (ntok * GRP + ROWS - 1) / ROWS + (int)nreq - 1;
    nsplit = std::max(1, std::min(64, sms / (rb * HK)));
  } else if (nsplit <= 0) {
    // 1 CTA / SM. Cost in tile-times: waves x tiles per split + the fp32 partial round trip of the merge
    // (ntok x HQ x D x 8 B per split; one tile-time moves ~26 KB per SM of DRAM traffic on both cards).
    // CTA count: exact for one request, + 1 partial row block per extra request.
    int nrb_est = (ntok * GRP + ROWS - 1) / ROWS + (int)nreq - 1;
    double best = 1e30;
    nsplit = 1;
    for (int sp = 1; sp <= 32 && (sp == 1 || max_tiles / sp >= 8) && sp * ntok <= 10922; ++sp) {
      double waves = std::ceil((double)nrb_est * HK * sp / sms), tps_ = std::ceil((double)max_tiles / sp);
      double cost = waves * tps_ + (sp > 1 ? sp * (double)ntok * HQ * D * 8 / (26624.0 * sms) + 2 : 0);
      if (cost < best) { best = cost; nsplit = sp; }
    }
  }
  int tps = 0;
  if (!graph_safe) {
    nsplit = std::max(1, std::min(nsplit, max_tiles));
    tps = (max_tiles + nsplit - 1) / nsplit;
    nsplit = (max_tiles + tps - 1) / tps;
  }
  torch::Tensor opart, mpart, lpart;
  if (nsplit > 1) {
    opart = torch::empty({nsplit, ntok, HQ, D}, opts_f);
    mpart = torch::empty({nsplit, ntok, HQ}, opts_f);
    lpart = torch::empty({nsplit, ntok, HQ}, opts_f);
  }
  Params p;
  p.q8 = q8.data_ptr<uint8_t>(); p.qsc = qsc.data_ptr<float>();
  p.kc = reinterpret_cast<const uint8_t*>(k_cache.data_ptr()); p.vc = reinterpret_cast<const uint8_t*>(v_cache.data_ptr());
  p.k_ps = k_cache.stride(0); p.k_ts = k_cache.stride(1); p.k_hs = k_cache.stride(2);
  p.v_ps = v_cache.stride(0); p.v_ts = v_cache.stride(1); p.v_hs = v_cache.stride(2);
  p.pt = page_table.data_ptr<int>(); p.pt_stride = page_table.stride(0);
  p.seq_lens = seq_lens.data_ptr<int>(); p.cu_q = cu_q.data_ptr<int>();
  p.nreq = (int)nreq; p.ntok = ntok; p.nsplit = nsplit; p.tiles_per_split = tps;
  p.v_out_scale = (float)v_descale;
  p.opart = nsplit > 1 ? opart.data_ptr<float>() : nullptr;
  p.mpart = nsplit > 1 ? mpart.data_ptr<float>() : nullptr;
  p.lpart = nsplit > 1 ? lpart.data_ptr<float>() : nullptr;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  size_t smem = NW * TQ * 256 + 2 * NST * KT * 256;   // NW x 4 KB + 64 KB
  static int attr_dev = -1;
  int dev = q.get_device();
  if (attr_dev != dev) {
    cudaFuncSetAttribute(fp8_extend_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_dev = dev;
  }
  fp8_extend_kernel<<<dim3(nrb, HK, nsplit), NT, smem, st>>>(p);
  if (nsplit > 1)
    merge_kernel<<<(rows * 32 + 255) / 256, 256, 0, st>>>(opart.data_ptr<float>(), mpart.data_ptr<float>(),
                                                          lpart.data_ptr<float>(), p.out, ntok, nsplit);
}

// single request convenience (tests): q [nq, HQ, D], page_table [width], total length L
torch::Tensor fp8_extend(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                         int64_t L, double k_descale, double v_descale, double sm_scale, int64_t nsplit_req) {
  auto out = torch::empty_like(q);
  auto opt = q.options().dtype(at::kInt);
  auto sl = torch::full({1}, (int)L, opt);
  auto cu = torch::tensor({0, (int)q.size(0)}, opt);
  fp8_extend_batch(q, k_cache, v_cache, page_table.view({1, -1}), sl, cu, 1, L, k_descale, v_descale, sm_scale, out,
                   nsplit_req);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_extend", &fp8_extend);
  m.def("fp8_extend_batch", &fp8_extend_batch);
}
