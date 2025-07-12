#include "common/common.h"
#include "common/tester.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include "common/print_data.hpp"

using namespace nvcuda;

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_SMEM_kernel(half *A, half *B,
                                                        half *C, int M, int N,
                                                        int K)
{
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int m_idx = threadIdx.y;
  const int n_idx = threadIdx.z;
  const int thread = threadIdx.x + blockDim.x * threadIdx.y +
                     blockDim.x * blockDim.y * threadIdx.z;
  const int share_mem_a_m = blockIdx.y * WMMA_M * WMMA_TILE_M;
  const int share_mem_b_n = blockIdx.x * WMMA_N * WMMA_TILE_N;

  const int load_gmem_a_m = share_mem_a_m + m_idx * WMMA_M;
  const int load_gmem_b_n = share_mem_b_n + n_idx * WMMA_N;
  static_assert(WMMA_N == WMMA_K);
  __shared__ half smem[WMMA_TILE_M + WMMA_TILE_N][WMMA_M * WMMA_N];

  static_assert(WMMA_TILE_M == 4);
  static_assert(WMMA_TILE_N == 2);

  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      A_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      B_frag;

  unsigned mmaIdx = thread % 4;
  unsigned mmaIdy = thread / 4;
  unsigned mmbIdx = thread % 16;
  unsigned mmbIdy = thread / 16;

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    auto aptr = A + share_mem_a_m * K + k * WMMA_K;
    auto bptr = B + (k * WMMA_K) * N + share_mem_b_n;

    reinterpret_cast<int64_t *>(smem)[thread] =
        reinterpret_cast<int64_t *>(aptr)[mmaIdx + mmaIdy * K / 4];

    reinterpret_cast<int *>(smem[WMMA_TILE_M])[thread] =
        reinterpret_cast<int *>(bptr)[mmbIdx + mmbIdy * N / 2];

    __syncthreads();

    wmma::load_matrix_sync(
        A_frag, reinterpret_cast<half *>(smem) + m_idx * WMMA_K * 16, WMMA_K);
    wmma::load_matrix_sync(
        B_frag, reinterpret_cast<half *>(smem[WMMA_TILE_M]) + n_idx * 16,
        WMMA_N * WMMA_TILE_N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    __syncthreads();
  }
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
                          wmma::mem_row_major);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int m_idx = threadIdx.y;
  const int n_idx = threadIdx.z;
  const int load_gmem_a_m = blockIdx.y * WMMA_M * 4 + m_idx * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N * 2 + n_idx * WMMA_N;
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
#if 0
    const int thread = threadIdx.x + blockDim.x * threadIdx.y +
                       blockDim.x * blockDim.y * threadIdx.z;
    if (threadIdx.y + threadIdx.z == 0 /*thread == 0*/) {
      printf("tid=(%d %d %d) "
             "A=(%f %f %f %f) "
             "B=(%f %f %f %f) "
             "C=(%f %f %f %f)\n",
             threadIdx.x, threadIdx.y, threadIdx.z, (float)A_frag.x[0],
             (float)A_frag.x[1], (float)A_frag.x[2], (float)A_frag.x[3],
             (float)B_frag.x[0], (float)B_frag.x[1], (float)B_frag.x[2],
             (float)B_frag.x[3], (float)C_frag.x[0], (float)C_frag.x[1],
             (float)C_frag.x[2], (float)C_frag.x[3]);
      // printf("aptr=(%f %f %f %f %f %f %f %f ) "
      //        "bptr=(%f %f %f %f %f %f %f %f ) "
      //        "\n",
      //        (float)A[0], (float)A[1], (float)A[2],
      //        (float)A[3], (float)A[4], (float)A[5], (float)A[6], (float)A[7],
      //        (float)B[0], (float)B[1], (float)B[2], (float)B[3], (float)B[4],
      //        (float)B[5], (float)B[6], (float)B[7]);
    }
#endif
    __syncthreads();
  }
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
                          wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_SME_mma4x2(half *A, half *B, half *C, int M, int N,
                                     int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  dim3 block_dim(32, WMMA_TILE_M, WMMA_TILE_N);
  dim3 grid_dim(div_ceil(N, WMMA_N * WMMA_TILE_N),
                div_ceil(M, WMMA_M * WMMA_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_SMEM_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N>
      <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

void hgemm_wmma_m16n16k16_mma4x2(half *A, half *B, half *C, int M, int N,
                                 int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  dim3 block_dim(32, WMMA_TILE_M, WMMA_TILE_N);
  dim3 grid_dim(div_ceil(N, WMMA_N * WMMA_TILE_N),
                div_ceil(M, WMMA_M * WMMA_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K>
      <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

int main() {
  constexpr unsigned M = 512;
  constexpr unsigned N = 2048;
  constexpr unsigned K = 1024;

  Tester tester(M, N, K, 1, 10, 100, true);
  tester.evaluate(hgemm_wmma_m16n16k16_SME_mma4x2,
                  "hgemm_wmma_m16n16k16_SME_mma4x2");

  printf(" ======================== \n");
  Tester tester2(M, N, K, 1, 10, 100, true);
  tester2.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");

  Matrix *c1 = tester.getC();
  Matrix *c2 = tester2.getC();

  int num = 0;
  for (size_t i = 0; i < c1->getSize(); i++) {
    if ((float)c1->getHostPtr()[i] - (float)c2->getHostPtr()[i] > 0.0001) {
      num++;
      printf("%d %f %f\n", (int)i, (float)c1->getHostPtr()[i],
             (float)c2->getHostPtr()[i]);
    }
    if (num > 10)
      break;
  }
  printf("num = %d\n", num);

  // PrintData2Txt(c1->getHostPtr(), "ref.txt", c1->getRow(), c1->getCol());
  // PrintData2Txt(c2->getHostPtr(), "res.txt", c2->getRow(), c2->getCol());
  // PrintData2Txt(tester.getA()->getHostPtr(), "a.txt", tester.getA()->getRow(),
  //               tester.getA()->getCol());
  // PrintData2Txt(tester.getB()->getHostPtr(), "b.txt", tester.getB()->getRow(),
  //               tester.getB()->getCol());

  return 0;
}