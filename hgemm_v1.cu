#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int load_gmem_a_m = blockIdx.y * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N;
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
}
// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_float_naive_kernel(half *A, half *B,
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

void hgemm_wmma_m16n16k16_naive(half *A, half *B, half *C, int M, int N,
                                int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  dim3 block_dim(32, 1, 1);
  dim3 grid_dim(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
  hgemm_wmma_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K>
      <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}

int main() {
  Tester tester(512, 2048, 1024, 1, 10, 100, true);
  tester.evaluate(hgemm_wmma_m16n16k16_naive, "hgemm_wmma_m16n16k16_naive");

  return 0;
}