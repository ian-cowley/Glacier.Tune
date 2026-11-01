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
// 8. Causal GQA Training Attention Forward (FlashAttention-2 Online Softmax)
// Supports arbitrary head_dim (1..1024) with warp-aligned blockDim.x = ceil(head_dim/32)*32.
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Dynamic Shared Memory: seq_len * sizeof(float) when probs != nullptr, else 0.
// =========================================================================
__global__ void tune_attn_causal_gqa_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    float* __restrict__ probs,
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

    __shared__ float s_warp_sum[32];
    extern __shared__ float s_scores[];

    size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float q_d = (tid < head_dim) ? q[q_offset] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int j = 0; j <= i; j++) {
        size_t kv_base = ((size_t)j * n_heads_kv + h_kv) * head_dim;
        float k_d = (tid < head_dim) ? k[kv_base + tid] : 0.0f;
        float v_d = (tid < head_dim) ? v[kv_base + tid] : 0.0f;

        float total_dot = block_reduce_sum(q_d * k_d, s_warp_sum, tid, num_warps);
        float s_t = total_dot * attn_scale;

        if (probs != nullptr && tid == 0) {
            s_scores[j] = s_t;
        }

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;
    }

    if (tid < head_dim) {
        size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
        attn_out[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
    }

    if (probs != nullptr) {
        __syncthreads();
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int j = tid; j <= i; j += blockDim.x) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            probs[p_offset] = expf(s_scores[j] - m) * inv_l;
        }
        for (int j = i + 1 + tid; j < seq_len; j += blockDim.x) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            probs[p_offset] = 0.0f;
        }
    }
}

// =========================================================================
// 9. Causal GQA Training Attention Backward: dQ and dS
// Supports arbitrary head_dim (1..1024) with warp-aligned blockDim.x = ceil(head_dim/32)*32.
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Dynamic Shared Memory: seq_len * sizeof(float).
// =========================================================================
__global__ void tune_attn_causal_gqa_bwd_dq_ds(
    const float* __restrict__ d_out,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ probs,
    float* __restrict__ dq,
    float* __restrict__ ds_out,
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

    __shared__ float s_warp_sum[32];
    extern __shared__ float s_dp[];

    size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float dout_d = (tid < head_dim) ? d_out[out_offset] : 0.0f;

    // 1. Compute dP_{ij} = sum_d dOut_{id} * V_{jd} for j <= i
    for (int j = 0; j <= i; j++) {
        float v_d = (tid < head_dim) ? v[((size_t)j * n_heads_kv + h_kv) * head_dim + tid] : 0.0f;
        float dp_j = block_reduce_sum(dout_d * v_d, s_warp_sum, tid, num_warps);
        if (tid == 0) {
            s_dp[j] = dp_j;
        }
    }
    __syncthreads();

    // 2. Compute dotPdP = sum_{j<=i} P_{ij} * dP_{ij}
    float my_dot = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        my_dot += probs[p_offset] * s_dp[j];
    }
    float dotPdP = block_reduce_sum(my_dot, s_warp_sum, tid, num_warps);

    // 3. Compute dS_{ij} and accumulate into dQ_{id} = sum_{j<=i} dS_{ij} * K_{jd}
    float acc_dq = 0.0f;
    for (int j = 0; j <= i; j++) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        float p_val = probs[p_offset];
        float ds_val = p_val * (s_dp[j] - dotPdP) * attn_scale;

        if (tid == 0) {
            ds_out[p_offset] = ds_val;
        }

        if (tid < head_dim) {
            float k_d = k[((size_t)j * n_heads_kv + h_kv) * head_dim + tid];
            acc_dq += ds_val * k_d;
        }
    }

    if (tid < head_dim) {
        dq[out_offset] = acc_dq;
    }
}

// =========================================================================
// 10. Causal GQA Training Attention Backward: dK and dV
// Grid: blockIdx.x = kv_h (0..n_heads_kv - 1), blockIdx.y = j (0..seq_len - 1)
// =========================================================================
__global__ void tune_attn_causal_gqa_bwd_dk_dv(
    const float* __restrict__ q,
    const float* __restrict__ d_out,
    const float* __restrict__ probs,
    const float* __restrict__ ds_buf,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim
) {
    int kv_h = blockIdx.x;
    int j = blockIdx.y;
    if (kv_h >= n_heads_kv || j >= seq_len) return;

    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int group_size = n_heads_q / n_heads_kv;
    int start_h = kv_h * group_size;
    int end_h = start_h + group_size;

    float acc_dk = 0.0f;
    float acc_dv = 0.0f;

    for (int h = start_h; h < end_h; h++) {
        for (int i = j; i < seq_len; i++) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            float ds_val = ds_buf[p_offset];
            float p_val = probs[p_offset];

            size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
            acc_dk += ds_val * q[q_offset];
            acc_dv += p_val * d_out[q_offset];
        }
    }

    size_t kv_offset = ((size_t)j * n_heads_kv + kv_h) * head_dim + tid;
    dk[kv_offset] = acc_dk;
    dv[kv_offset] = acc_dv;
}

} // extern "C"
