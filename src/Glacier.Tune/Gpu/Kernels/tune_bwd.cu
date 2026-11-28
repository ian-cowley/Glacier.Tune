// Glacier.Tune High-Performance CUDA Training Kernels
// Provides transposed quantized & FP32 backward GEMM kernels (dX += dY * W)
// plus F32 / Q4_0 forward batch GEMMs for complete type parity in VRAM.
//
// Target Architectures:
//   sm_75, sm_80, sm_86, sm_89, sm_90 + compute_75 PTX forward-compatibility.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define QK_K 256
#define QK8_0 32
#define QK4_0 32
#define WARP_SIZE 32

#pragma pack(push, 1)
struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};

struct BlockQ4_0 {
    half d;
    uint8_t qs[16];
};

struct BlockQ4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
#pragma pack(pop)

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* s_warp_sum, int tid, int num_warps) {
    val = warp_reduce_sum(val);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_warp_sum[warp_id] = val;
    }
    __syncthreads();

    float total = 0.0f;
    for (int w = 0; w < num_warps; w++) {
        total += s_warp_sum[w];
    }
    __syncthreads();
    return total;
}

extern "C" {

// =========================================================================
// 1. Batched Forward GEMM for F32: Y[batch_size, m_rows] = X[batch_size, k_cols] * W^T
// =========================================================================
__global__ void gemm_f32_batch(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const float* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = threadIdx.x / WARP_SIZE; // 0..3
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31
    int row = blockIdx.x * 4 + warp_id;
    int t_base = blockIdx.y * 8;

    if (row >= m_rows || t_base >= batch_size) return;

    const float* row_w = W + (size_t)row * k_cols;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int c = lane_id; c < k_cols; c += WARP_SIZE) {
        float w = row_w[c];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int t = t_base + i;
            if (t < batch_size) {
                acc[i] += w * X[(size_t)t * k_cols + c];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int t = t_base + i;
        if (t < batch_size) {
            float sum = warp_reduce_sum(acc[i]);
            if (lane_id == 0) {
                if (bias != nullptr) sum += bias[row];
                if (residual != nullptr) residual[(size_t)t * m_rows + row] += sum;
                if (Y != nullptr) Y[(size_t)t * m_rows + row] = sum;
            }
        }
    }
}

// =========================================================================
// 2. Batched Forward GEMM for Q4_0: Y[batch_size, m_rows] = X[batch_size, k_cols] * W^T
// =========================================================================
__global__ void gemm_q4_0_batch(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const BlockQ4_0* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * 4 + warp_id;
    int t_base = blockIdx.y * 8;

    if (row >= m_rows || t_base >= batch_size) return;

    int nb = k_cols / QK4_0;
    const BlockQ4_0* row_w = W + (size_t)row * nb;
    int qs_idx = lane_id & 15;
    int shift = (lane_id >> 4) << 2;

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ4_0* blk = &row_w[b];
        float d = __half2float(blk->d);
        int q = ((int)(blk->qs[qs_idx] >> shift) & 0x0F) - 8;
        float w = d * (float)q;
        int col = b * QK4_0 + lane_id;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int t = t_base + i;
            if (t < batch_size) {
                acc[i] += w * X[(size_t)t * k_cols + col];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int t = t_base + i;
        if (t < batch_size) {
            float sum = warp_reduce_sum(acc[i]);
            if (lane_id == 0) {
                if (bias != nullptr) sum += bias[row];
                if (residual != nullptr) residual[(size_t)t * m_rows + row] += sum;
                if (Y != nullptr) Y[(size_t)t * m_rows + row] = sum;
            }
        }
    }
}

// =========================================================================
// 3. Batched Backward Transpose-GEMM for Q4_K: dX[batch_size, k_cols] += dY[batch_size, m_rows] * W
// blockDim.x = 256 (1 thread per column in Q4_K super-block), gridDim.x = k_cols / 256, gridDim.y = ceil(batch_size / 8)
// =========================================================================
__global__ void gemm_q4_k_bwd(
    float* __restrict__ dX,
    const float* __restrict__ dY,
    const BlockQ4_K* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int sb = blockIdx.x;
    int c_local = threadIdx.x; // 0..255
    int col = sb * QK_K + c_local;
    int t_base = blockIdx.y * 8;

    if (col >= k_cols || t_base >= batch_size) return;

    int nb = k_cols / QK_K;
    int ib = c_local >> 5;                      // 0..7
    int qs_idx = ((ib >> 1) << 5) + (c_local & 31); // 0..127
    int shift = (ib & 1) << 2;                  // 0 or 4

    __shared__ float s_dy[8][16];

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int r_base = 0; r_base < m_rows; r_base += 16) {
        int r_count = m_rows - r_base;
        if (r_count > 16) r_count = 16;

        // Cooperatively load dY tile [8, 16] into shared memory (128 elements)
        if (c_local < 128) {
            int ti = c_local >> 4;   // 0..7
            int ri = c_local & 15;   // 0..15
            int t = t_base + ti;
            int r = r_base + ri;
            s_dy[ti][ri] = (t < batch_size && r < m_rows) ? dY[(size_t)t * m_rows + r] : 0.0f;
        }
        __syncthreads();

        for (int ri = 0; ri < r_count; ri++) {
            const BlockQ4_K* blk = &W[(size_t)(r_base + ri) * nb + sb];
            uint8_t sc, m;
            get_scale_min_k4(ib, blk->scales, &sc, &m);
            float d_sub = __half2float(blk->d) * (float)sc;
            float m_sub = __half2float(blk->dmin) * (float)m;
            uint8_t qnib = (blk->qs[qs_idx] >> shift) & 0x0F;
            float w = d_sub * (float)qnib - m_sub;

            #pragma unroll
            for (int ti = 0; ti < 8; ti++) {
                acc[ti] += s_dy[ti][ri] * w;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ti = 0; ti < 8; ti++) {
        int t = t_base + ti;
        if (t < batch_size) {
            dX[(size_t)t * k_cols + col] += acc[ti];
        }
    }
}

// =========================================================================
// 4. Batched Backward Transpose-GEMM for Q6_K: dX[batch_size, k_cols] += dY[batch_size, m_rows] * W
// blockDim.x = 256, gridDim.x = k_cols / 256, gridDim.y = ceil(batch_size / 8)
// =========================================================================
__global__ void gemm_q6_k_bwd(
    float* __restrict__ dX,
    const float* __restrict__ dY,
    const BlockQ6_K* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int sb = blockIdx.x;
    int c_local = threadIdx.x; // 0..255
    int col = sb * QK_K + c_local;
    int t_base = blockIdx.y * 8;

    if (col >= k_cols || t_base >= batch_size) return;

    int nb = k_cols / QK_K;
    int step = c_local >> 7;        // 0..1
    int rem = c_local & 127;        // 0..127
    int quad = rem >> 5;            // 0..3
    int lane = rem & 31;            // 0..31
    int is_idx = lane >> 4;         // 0..1

    int qh_idx = step * 32 + lane;
    int ql_idx = step * 64 + (quad & 1) * 32 + lane;
    int ql_shift = (quad >= 2) ? 4 : 0;
    int qh_shift = quad * 2;
    int sc_idx = step * 8 + is_idx + quad * 2;

    __shared__ float s_dy[8][16];

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int r_base = 0; r_base < m_rows; r_base += 16) {
        int r_count = m_rows - r_base;
        if (r_count > 16) r_count = 16;

        if (c_local < 128) {
            int ti = c_local >> 4;
            int ri = c_local & 15;
            int t = t_base + ti;
            int r = r_base + ri;
            s_dy[ti][ri] = (t < batch_size && r < m_rows) ? dY[(size_t)t * m_rows + r] : 0.0f;
        }
        __syncthreads();

        for (int ri = 0; ri < r_count; ri++) {
            const BlockQ6_K* blk = &W[(size_t)(r_base + ri) * nb + sb];
            uint8_t ql_nib = (blk->ql[ql_idx] >> ql_shift) & 0x0F;
            uint8_t qh_bits = (blk->qh[qh_idx] >> qh_shift) & 0x03;
            int q = (int)(ql_nib | (qh_bits << 4)) - 32;
            float w = __half2float(blk->d) * (float)blk->scales[sc_idx] * (float)q;

            #pragma unroll
            for (int ti = 0; ti < 8; ti++) {
                acc[ti] += s_dy[ti][ri] * w;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ti = 0; ti < 8; ti++) {
        int t = t_base + ti;
        if (t < batch_size) {
            dX[(size_t)t * k_cols + col] += acc[ti];
        }
    }
}

// =========================================================================
// 5. Batched Backward Transpose-GEMM for Q8_0: dX[batch_size, k_cols] += dY[batch_size, m_rows] * W
// blockDim.x = 128 (4 Q8_0 blocks per CTA), gridDim.x = ceil(k_cols / 128), gridDim.y = ceil(batch_size / 8)
// =========================================================================
__global__ void gemm_q8_0_bwd(
    float* __restrict__ dX,
    const float* __restrict__ dY,
    const BlockQ8_0* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int tid = threadIdx.x; // 0..127
    int col = blockIdx.x * 128 + tid;
    int t_base = blockIdx.y * 8;

    if (t_base >= batch_size) return;

    int nb = k_cols / QK8_0;
    int b = col >> 5;
    int elem = col & 31;

    __shared__ float s_dy[8][16];

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int r_base = 0; r_base < m_rows; r_base += 16) {
        int r_count = m_rows - r_base;
        if (r_count > 16) r_count = 16;

        int ti_load = tid >> 4;   // 0..7
        int ri_load = tid & 15;   // 0..15
        int t_load = t_base + ti_load;
        int r_load = r_base + ri_load;
        s_dy[ti_load][ri_load] = (t_load < batch_size && r_load < m_rows) ? dY[(size_t)t_load * m_rows + r_load] : 0.0f;
        __syncthreads();

        if (col < k_cols) {
            for (int ri = 0; ri < r_count; ri++) {
                const BlockQ8_0* blk = &W[(size_t)(r_base + ri) * nb + b];
                float w = __half2float(blk->d) * (float)blk->qs[elem];
                #pragma unroll
                for (int ti = 0; ti < 8; ti++) {
                    acc[ti] += s_dy[ti][ri] * w;
                }
            }
        }
        __syncthreads();
    }

    if (col < k_cols) {
        #pragma unroll
        for (int ti = 0; ti < 8; ti++) {
            int t = t_base + ti;
            if (t < batch_size) {
                dX[(size_t)t * k_cols + col] += acc[ti];
            }
        }
    }
}

// =========================================================================
// 6. Batched Backward Transpose-GEMM for Q4_0: dX[batch_size, k_cols] += dY[batch_size, m_rows] * W
// blockDim.x = 128, gridDim.x = ceil(k_cols / 128), gridDim.y = ceil(batch_size / 8)
// =========================================================================
__global__ void gemm_q4_0_bwd(
    float* __restrict__ dX,
    const float* __restrict__ dY,
    const BlockQ4_0* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int tid = threadIdx.x; // 0..127
    int col = blockIdx.x * 128 + tid;
    int t_base = blockIdx.y * 8;

    if (t_base >= batch_size) return;

    int nb = k_cols / QK4_0;
    int b = col >> 5;
    int elem = col & 31;
    int qs_idx = elem & 15;
    int shift = (elem >> 4) << 2;

    __shared__ float s_dy[8][16];

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int r_base = 0; r_base < m_rows; r_base += 16) {
        int r_count = m_rows - r_base;
        if (r_count > 16) r_count = 16;

        int ti_load = tid >> 4;
        int ri_load = tid & 15;
        int t_load = t_base + ti_load;
        int r_load = r_base + ri_load;
        s_dy[ti_load][ri_load] = (t_load < batch_size && r_load < m_rows) ? dY[(size_t)t_load * m_rows + r_load] : 0.0f;
        __syncthreads();

        if (col < k_cols) {
            for (int ri = 0; ri < r_count; ri++) {
                const BlockQ4_0* blk = &W[(size_t)(r_base + ri) * nb + b];
                int q = ((int)(blk->qs[qs_idx] >> shift) & 0x0F) - 8;
                float w = __half2float(blk->d) * (float)q;
                #pragma unroll
                for (int ti = 0; ti < 8; ti++) {
                    acc[ti] += s_dy[ti][ri] * w;
                }
            }
        }
        __syncthreads();
    }

    if (col < k_cols) {
        #pragma unroll
        for (int ti = 0; ti < 8; ti++) {
            int t = t_base + ti;
            if (t < batch_size) {
                dX[(size_t)t * k_cols + col] += acc[ti];
            }
        }
    }
}

// =========================================================================
// 7. Batched Backward Transpose-GEMM for F32: dX[batch_size, k_cols] += dY[batch_size, m_rows] * W
// blockDim.x = 128, gridDim.x = ceil(k_cols / 128), gridDim.y = ceil(batch_size / 8)
// =========================================================================
__global__ void gemm_f32_bwd(
    float* __restrict__ dX,
    const float* __restrict__ dY,
    const float* __restrict__ W,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int tid = threadIdx.x; // 0..127
    int col = blockIdx.x * 128 + tid;
    int t_base = blockIdx.y * 8;

    if (t_base >= batch_size) return;

    __shared__ float s_dy[8][16];

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int r_base = 0; r_base < m_rows; r_base += 16) {
        int r_count = m_rows - r_base;
        if (r_count > 16) r_count = 16;

        int ti_load = tid >> 4;
        int ri_load = tid & 15;
        int t_load = t_base + ti_load;
        int r_load = r_base + ri_load;
        s_dy[ti_load][ri_load] = (t_load < batch_size && r_load < m_rows) ? dY[(size_t)t_load * m_rows + r_load] : 0.0f;
        __syncthreads();

        if (col < k_cols) {
            for (int ri = 0; ri < r_count; ri++) {
                float w = W[(size_t)(r_base + ri) * k_cols + col];
                #pragma unroll
                for (int ti = 0; ti < 8; ti++) {
                    acc[ti] += s_dy[ti][ri] * w;
                }
            }
        }
        __syncthreads();
    }

    if (col < k_cols) {
        #pragma unroll
        for (int ti = 0; ti < 8; ti++) {
            int t = t_base + ti;
            if (t < batch_size) {
                dX[(size_t)t * k_cols + col] += acc[ti];
            }
        }
    }
}

// =========================================================================
// 8. Causal GQA Training Attention Forward (FlashAttention-2 K/V Tiled + Online Softmax)
// Supports arbitrary head_dim (1..1024) and UNBOUNDED seq_len via K/V shared-memory tiling.
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Dynamic Shared Memory: 2 * bc * head_dim * sizeof(float), where
//   bc = (head_dim <= 128) ? 32 : ((head_dim <= 256) ? 16 : ((head_dim <= 512) ? 8 : 4))
// Stores O(H * S) LSE[h, i] = m + log(l) instead of O(H * S^2) probs.
// =========================================================================
__device__ __forceinline__ int attn_tile_bc(int head_dim) {
    if (head_dim <= 128) return 32;
    if (head_dim <= 256) return 16;
    if (head_dim <= 512) return 8;
    return 4;
}

__global__ void tune_attn_causal_gqa_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    float* __restrict__ lse_out,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float attn_scale
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads_q || i >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    int bc = attn_tile_bc(head_dim);
    __shared__ float s_warp_sum[32];
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = smem + (size_t)bc * head_dim;

    size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float q_d = (tid < head_dim) ? q[q_offset] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int j_start = 0; j_start <= i; j_start += bc) {
        int tile_len = min(bc, i - j_start + 1);
        int total_elems = tile_len * head_dim;

        for (int idx = tid; idx < total_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int j = j_start + t;
            size_t kv_off = ((size_t)j * n_heads_kv + h_kv) * head_dim + d;
            s_k[idx] = k[kv_off];
            s_v[idx] = v[kv_off];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float k_d = (tid < head_dim) ? s_k[(size_t)t * head_dim + tid] : 0.0f;
            float v_d = (tid < head_dim) ? s_v[(size_t)t * head_dim + tid] : 0.0f;

            float total_dot = block_reduce_sum(q_d * k_d, s_warp_sum, tid, num_warps);
            float s_t = total_dot * attn_scale;

            float m_new = fmaxf(m, s_t);
            float alpha = (l > 0.0f) ? expf(m - m_new) : 0.0f;
            float w_t = expf(s_t - m_new);

            acc = acc * alpha + w_t * v_d;
            l = l * alpha + w_t;
            m = m_new;
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
        attn_out[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
    }

    if (lse_out != nullptr && tid == 0) {
        lse_out[(size_t)h * seq_len + i] = (l > 0.0f) ? (m + logf(l)) : -1e30f;
    }
}

// =========================================================================
// 9. FlashAttention-2 Preprocess D[h, i] = sum_d dOut[i, h, d] * Out[i, h, d]
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// =========================================================================
__global__ void tune_attn_bwd_preprocess_d(
    const float* __restrict__ d_out,
    const float* __restrict__ attn_out,
    float* __restrict__ d_buf,
    int seq_len,
    int n_heads_q,
    int head_dim
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads_q || i >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    __shared__ float s_warp_sum[32];

    size_t offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float prod = (tid < head_dim) ? (d_out[offset] * attn_out[offset]) : 0.0f;
    float total = block_reduce_sum(prod, s_warp_sum, tid, num_warps);
    if (tid == 0) {
        d_buf[(size_t)h * seq_len + i] = total;
    }
}

// =========================================================================
// 10. Causal GQA FlashAttention-2 Backward: dQ (LSE recompute, K/V shared tiling)
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Dynamic Shared Memory: 2 * bc * head_dim * sizeof(float)
// =========================================================================
__global__ void tune_attn_causal_gqa_bwd_dq(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ d_out,
    const float* __restrict__ lse,
    const float* __restrict__ d_buf,
    float* __restrict__ dq,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float attn_scale
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads_q || i >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    int bc = attn_tile_bc(head_dim);
    __shared__ float s_warp_sum[32];
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = smem + (size_t)bc * head_dim;

    size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float q_d = (tid < head_dim) ? q[out_offset] : 0.0f;
    float dout_d = (tid < head_dim) ? d_out[out_offset] : 0.0f;
    float lse_i = lse[(size_t)h * seq_len + i];
    float d_i = d_buf[(size_t)h * seq_len + i];

    float acc_dq = 0.0f;
    for (int j_start = 0; j_start <= i; j_start += bc) {
        int tile_len = min(bc, i - j_start + 1);
        int total_elems = tile_len * head_dim;

        for (int idx = tid; idx < total_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int j = j_start + t;
            size_t kv_off = ((size_t)j * n_heads_kv + h_kv) * head_dim + d;
            s_k[idx] = k[kv_off];
            s_v[idx] = v[kv_off];
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float k_d = (tid < head_dim) ? s_k[(size_t)t * head_dim + tid] : 0.0f;
            float v_d = (tid < head_dim) ? s_v[(size_t)t * head_dim + tid] : 0.0f;

            float s_t = block_reduce_sum(q_d * k_d, s_warp_sum, tid, num_warps) * attn_scale;
            float dp_t = block_reduce_sum(dout_d * v_d, s_warp_sum, tid, num_warps);
            float p_t = expf(s_t - lse_i);
            float ds_t = p_t * (dp_t - d_i) * attn_scale;

            if (tid < head_dim) {
                acc_dq += ds_t * k_d;
            }
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        dq[out_offset] = acc_dq;
    }
}

// =========================================================================
// 11. Causal GQA FlashAttention-2 Backward: dK and dV (LSE recompute, Q/dOut shared tiling)
// Grid: blockIdx.x = kv_h (0..n_heads_kv - 1), blockIdx.y = j (0..seq_len - 1)
// Dynamic Shared Memory: 2 * bc * head_dim * sizeof(float)
// =========================================================================
__global__ void tune_attn_causal_gqa_bwd_dk_dv(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ d_out,
    const float* __restrict__ lse,
    const float* __restrict__ d_buf,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float attn_scale
) {
    int kv_h = blockIdx.x;
    int j = blockIdx.y;
    if (kv_h >= n_heads_kv || j >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    int group_size = n_heads_q / n_heads_kv;
    int start_h = kv_h * group_size;
    int end_h = start_h + group_size;

    int bc = attn_tile_bc(head_dim);
    __shared__ float s_warp_sum[32];
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_dout = smem + (size_t)bc * head_dim;

    size_t kv_offset = ((size_t)j * n_heads_kv + kv_h) * head_dim + tid;
    float k_d = (tid < head_dim) ? k[kv_offset] : 0.0f;
    float v_d = (tid < head_dim) ? v[kv_offset] : 0.0f;

    float acc_dk = 0.0f;
    float acc_dv = 0.0f;

    for (int h = start_h; h < end_h; h++) {
        for (int i_start = j; i_start < seq_len; i_start += bc) {
            int tile_len = min(bc, seq_len - i_start);
            int total_elems = tile_len * head_dim;

            for (int idx = tid; idx < total_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int d = idx - t * head_dim;
                int i = i_start + t;
                size_t q_off = ((size_t)i * n_heads_q + h) * head_dim + d;
                s_q[idx] = q[q_off];
                s_dout[idx] = d_out[q_off];
            }
            __syncthreads();

            for (int t = 0; t < tile_len; t++) {
                int i = i_start + t;
                float q_d = (tid < head_dim) ? s_q[(size_t)t * head_dim + tid] : 0.0f;
                float dout_d = (tid < head_dim) ? s_dout[(size_t)t * head_dim + tid] : 0.0f;

                float s_t = block_reduce_sum(q_d * k_d, s_warp_sum, tid, num_warps) * attn_scale;
                float dp_t = block_reduce_sum(dout_d * v_d, s_warp_sum, tid, num_warps);
                float lse_i = lse[(size_t)h * seq_len + i];
                float d_i = d_buf[(size_t)h * seq_len + i];

                float p_t = expf(s_t - lse_i);
                float ds_t = p_t * (dp_t - d_i) * attn_scale;

                if (tid < head_dim) {
                    acc_dk += ds_t * q_d;
                    acc_dv += p_t * dout_d;
                }
            }
            __syncthreads();
        }
    }

    if (tid < head_dim) {
        dk[kv_offset] = acc_dk;
        dv[kv_offset] = acc_dv;
    }
}

// =========================================================================
// 12. Resident RMSNorm Backward, Vector Residual / Scale Helpers
// =========================================================================
__global__ void tune_rmsnorm_bwd(
    const float* __restrict__ d_out,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ d_x,
    int seq_len,
    int dim,
    float eps
) {
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    __shared__ float s_warp_sum[32];

    size_t row_off = (size_t)t * dim;
    float local_sq = 0.0f;
    float local_dot = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float xv = x[row_off + i];
        float dy = d_out[row_off + i] * w[i];
        local_sq += xv * xv;
        local_dot += dy * xv;
    }

    float sum_sq = block_reduce_sum(local_sq, s_warp_sum, tid, num_warps);
    float sum_dot = block_reduce_sum(local_dot, s_warp_sum, tid, num_warps);

    float inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    float scale2 = sum_dot * (inv_rms * inv_rms * inv_rms) / (float)dim;

    for (int i = tid; i < dim; i += blockDim.x) {
        float dy = d_out[row_off + i] * w[i];
        d_x[row_off + i] = dy * inv_rms - x[row_off + i] * scale2;
    }
}

__global__ void tune_res_add_out(
    float* __restrict__ dst,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = a[idx] + b[idx];
    }
}

__global__ void tune_scale_buf(
    float* __restrict__ buf,
    float scale,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        buf[idx] *= scale;
    }
}

// =========================================================================
// 13. GPU-Resident LoRA Forward & Backward Kernels
// Matches Glacier.Tune LoraAdapter layout: A: [in_f, rank], B: [rank, out_f], low_rank: [seq_len, rank]
// =========================================================================
__global__ void tune_lora_proj_lr(
    const float* __restrict__ X,
    const float* __restrict__ A,
    float* __restrict__ low_rank,
    int seq_len,
    int in_f,
    int rank
) {
    int r = blockIdx.x;
    int t = blockIdx.y;
    if (r >= rank || t >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    __shared__ float s_warp_sum[32];

    const float* x_row = X + (size_t)t * in_f;
    float local_sum = 0.0f;
    for (int k = tid; k < in_f; k += blockDim.x) {
        local_sum += x_row[k] * A[(size_t)k * rank + r];
    }
    float total = block_reduce_sum(local_sum, s_warp_sum, tid, num_warps);
    if (tid == 0) {
        low_rank[(size_t)t * rank + r] = total;
    }
}

__global__ void tune_lora_add_b(
    float* __restrict__ Y,
    const float* __restrict__ low_rank,
    const float* __restrict__ B,
    int seq_len,
    int out_f,
    int rank,
    float scaling
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (row >= out_f || t >= seq_len) return;

    const float* lr_row = low_rank + (size_t)t * rank;
    float acc = 0.0f;
    for (int r = 0; r < rank; r++) {
        acc += lr_row[r] * B[(size_t)r * out_f + row];
    }
    Y[(size_t)t * out_f + row] += scaling * acc;
}

__global__ void tune_lora_bwd_db(
    const float* __restrict__ low_rank,
    const float* __restrict__ dY,
    float* __restrict__ gradB,
    int seq_len,
    int out_f,
    int rank,
    float scaling
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (row >= out_f || r >= rank) return;

    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        acc += dY[(size_t)t * out_f + row] * low_rank[(size_t)t * rank + r];
    }
    gradB[(size_t)r * out_f + row] += scaling * acc;
}

__global__ void tune_lora_bwd_dlr(
    const float* __restrict__ dY,
    const float* __restrict__ B,
    float* __restrict__ dLR,
    int seq_len,
    int out_f,
    int rank,
    float scaling
) {
    int r = blockIdx.x;
    int t = blockIdx.y;
    if (r >= rank || t >= seq_len) return;

    int tid = threadIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    __shared__ float s_warp_sum[32];

    const float* dy_row = dY + (size_t)t * out_f;
    const float* b_row = B + (size_t)r * out_f;
    float local_sum = 0.0f;
    for (int row = tid; row < out_f; row += blockDim.x) {
        local_sum += dy_row[row] * b_row[row];
    }
    float total = block_reduce_sum(local_sum, s_warp_sum, tid, num_warps);
    if (tid == 0) {
        dLR[(size_t)t * rank + r] = scaling * total;
    }
}

__global__ void tune_lora_bwd_da(
    const float* __restrict__ X,
    const float* __restrict__ dLR,
    float* __restrict__ gradA,
    int seq_len,
    int in_f,
    int rank
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)in_f * rank;
    if (idx >= total) return;

    int k = (int)(idx / (size_t)rank);
    int r = (int)(idx - (size_t)k * rank);

    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        acc += X[(size_t)t * in_f + k] * dLR[(size_t)t * rank + r];
    }
    gradA[idx] += acc;
}

__global__ void tune_lora_bwd_dx(
    const float* __restrict__ dLR,
    const float* __restrict__ A,
    float* __restrict__ dX,
    int seq_len,
    int in_f,
    int rank
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (k >= in_f || t >= seq_len) return;

    const float* dlr_row = dLR + (size_t)t * rank;
    const float* a_row = A + (size_t)k * rank;
    float acc = 0.0f;
    for (int r = 0; r < rank; r++) {
        acc += dlr_row[r] * a_row[r];
    }
    dX[(size_t)t * in_f + k] += acc;
}

// =========================================================================
// 14. GPU-Resident Tiled Softmax Cross-Entropy (Barron Log1p Split-Sum)
// Keeps [valid_count, vocab_tile] logits 100% resident in VRAM.
// =========================================================================
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void tune_ce_tile_stats(
    const float* __restrict__ tile_logits,
    const int* __restrict__ target_ids,
    int v_start,
    int v_count,
    int valid_count,
    float* __restrict__ running_max,
    int* __restrict__ running_argmax,
    double* __restrict__ running_sum_other,
    float* __restrict__ target_logits
) {
    int i = blockIdx.x;
    if (i >= valid_count) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float s_warp_max[32];
    __shared__ int s_warp_idx[32];
    __shared__ double s_warp_sum[32];
    __shared__ float s_tile_max;
    __shared__ int s_tile_argmax;

    const float* row = tile_logits + (size_t)i * v_count;
    float local_max = -1e38f;
    int local_idx = -1;
    for (int j = tid; j < v_count; j += blockDim.x) {
        float val = row[j];
        int g = v_start + j;
        if (val > local_max || (val == local_max && (local_idx < 0 || g < local_idx))) {
            local_max = val;
            local_idx = g;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float o_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int o_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (o_val > local_max || (o_val == local_max && o_idx >= 0 && (local_idx < 0 || o_idx < local_idx))) {
            local_max = o_val;
            local_idx = o_idx;
        }
    }

    if (lane_id == 0) {
        s_warp_max[warp_id] = local_max;
        s_warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float m = -1e38f;
        int idx = -1;
        for (int w = 0; w < num_warps; w++) {
            float wm = s_warp_max[w];
            int wi = s_warp_idx[w];
            if (wi >= 0 && (wm > m || (wm == m && (idx < 0 || wi < idx)))) {
                m = wm;
                idx = wi;
            }
        }
        s_tile_max = m;
        s_tile_argmax = idx;

        int target = target_ids[i];
        if (v_start == 0) {
            target_logits[i] = 0.0f;
        }
        if (target >= v_start && target < v_start + v_count) {
            target_logits[i] = row[target - v_start];
        }
    }
    __syncthreads();

    float tile_max = s_tile_max;
    int tile_argmax = s_tile_argmax;

    double local_sum = 0.0;
    for (int j = tid; j < v_count; j += blockDim.x) {
        int g = v_start + j;
        if (g != tile_argmax) {
            local_sum += exp((double)row[j] - (double)tile_max);
        }
    }

    local_sum = warp_reduce_sum_f64(local_sum);
    if (lane_id == 0) {
        s_warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        double tile_sum_other = 0.0;
        for (int w = 0; w < num_warps; w++) {
            tile_sum_other += s_warp_sum[w];
        }

        if (v_start == 0) {
            running_max[i] = tile_max;
            running_argmax[i] = tile_argmax;
            running_sum_other[i] = tile_sum_other;
        } else {
            float prev_max = running_max[i];
            double prev_sum = running_sum_other[i];
            if (tile_max > prev_max) {
                running_sum_other[i] = tile_sum_other + exp((double)prev_max - (double)tile_max) * (1.0 + prev_sum);
                running_max[i] = tile_max;
                running_argmax[i] = tile_argmax;
            } else {
                running_sum_other[i] = prev_sum + exp((double)tile_max - (double)prev_max) * (1.0 + tile_sum_other);
            }
        }
    }
}

__global__ void tune_ce_finalize_loss(
    const float* __restrict__ running_max,
    const double* __restrict__ running_sum_other,
    const float* __restrict__ target_logits,
    double* __restrict__ inv_denom_out,
    float* __restrict__ loss_out,
    int valid_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= valid_count) return;

    double m = (double)running_max[i];
    double s = running_sum_other[i];
    double log_sum_exp = m + log1p(s);
    loss_out[i] = (float)(log_sum_exp - (double)target_logits[i]);
    inv_denom_out[i] = 1.0 / (1.0 + s);
}

__global__ void tune_ce_write_grad(
    float* __restrict__ tile_logits,
    const int* __restrict__ target_ids,
    int v_start,
    int v_count,
    int valid_count,
    const float* __restrict__ running_max,
    const int* __restrict__ running_argmax,
    const double* __restrict__ running_sum_other,
    const double* __restrict__ inv_denom_arr,
    float grad_scale
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (j >= v_count || i >= valid_count) return;

    size_t idx = (size_t)i * v_count + j;
    int g = v_start + j;
    int max_idx = running_argmax[i];
    int target = target_ids[i];
    double inv_denom = inv_denom_arr[i];

    double grad;
    if (g == max_idx) {
        if (g == target) {
            grad = (-running_sum_other[i] * inv_denom) * (double)grad_scale;
        } else {
            grad = inv_denom * (double)grad_scale;
        }
    } else {
        double p = exp((double)tile_logits[idx] - (double)running_max[i]) * inv_denom;
        grad = ((g == target) ? (p - 1.0) : p) * (double)grad_scale;
    }

    tile_logits[idx] = (float)grad;
}

} // extern "C"
