#include "common/common.h"
#include "common/tester.h"
#include <cuda_runtime.h>
#include <iostream>
using namespace nvcuda;

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int m_idx = threadIdx.y;
  const int n_idx = threadIdx.z;
  const int load_gmem_a_m = blockIdx.y * WMMA_M * 4 + m_idx;
  const int load_gmem_b_n = blockIdx.x * WMMA_N * 2 + n_idx;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      A_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      B_frag;

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
    wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads();
  }
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
                          wmma::mem_row_major);
  if (blockIdx.x + blockIdx.y + blockIdx.z + threadIdx.x + threadIdx.y +
          threadIdx.z ==
      0) {
    printf("C_frag: %f %f %f %f\n", (float)C_frag.x[0], (float)C_frag.x[1], (float)C_frag.x[2],
           (float)C_frag.x[3]);
    }
}
// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_float_mma4x2_kernel(half *A, half *B,
                                                        half *C, int M, int N,
                                                        int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int load_gmem_a_m = blockIdx.y * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
  wmma::fill_fragment(C_frag, 0.0f); // 注意使用0.0f

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag;

    wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
    wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads();
  }

  // 修改2: 将float累加器转换为half输出
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag_half;
#pragma unroll
  for (int i = 0; i < C_frag.num_elements; ++i) {
    C_frag_half.x[i] = __float2half(C_frag.x[i]); // 显式类型转换
  }
  // 存储转换后的half结果
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag_half, N,
                          wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_mma4x2(half *A, half *B, half *C, int M, int N,
                                int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  dim3 block_dim(32, WMMA_TILE_M, WMMA_TILE_N);
  dim3 grid_dim(div_ceil(N, WMMA_N * WMMA_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M));
  hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K>
      <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

int main() {
  Tester tester(512, 2048, 1024, 1, 10, 100, true);
  tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");

  return 0;
}