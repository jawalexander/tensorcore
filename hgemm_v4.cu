#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/print_data.hpp"
#include "common/tester.h"

using namespace nvcuda;

#define warpSize 32

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma8X8_kernel(half* A, half* B, half* C, int M, int N, int K)
{
    const int     NUM_K_TILES = div_ceil(K, WMMA_K);
    const int     m_idx       = threadIdx.y;
    const int     n_idx       = threadIdx.z;
    const int     thread      = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    constexpr int threadNum   = WMMA_TILE_M * WMMA_TILE_N * warpSize;

    constexpr int WARP_TILE   = 8;
    constexpr int WARP_TILE_M = WARP_TILE / WMMA_TILE_M;
    constexpr int WARP_TILE_N = WARP_TILE / WMMA_TILE_N;

    const int share_mem_a_m = blockIdx.y * WARP_TILE * WMMA_M;
    const int share_mem_b_n = blockIdx.x * WARP_TILE * WMMA_M;

    const int load_gmem_a_m = share_mem_a_m + m_idx * WMMA_M * WARP_TILE_M;
    const int load_gmem_b_n = share_mem_b_n + n_idx * WMMA_N * WARP_TILE_N;
    static_assert(WMMA_N == WMMA_K);
    __shared__ half smem[2][WARP_TILE][WMMA_M * WMMA_N];

    static_assert(WMMA_TILE_M == 4);
    static_assert(WMMA_TILE_N == 2);

    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M][WARP_TILE_N];
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag[WARP_TILE_N];

    unsigned mmaIdx = thread % 4;
    unsigned mmaIdy = thread / 4;
    unsigned mmbIdx = thread % 32;
    unsigned mmbIdy = thread / 32;

    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        auto aptr = A + share_mem_a_m * K + k * WMMA_K;
        auto bptr = B + (k * WMMA_K) * N + share_mem_b_n;

#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            reinterpret_cast<int64_t*>(smem[0])[thread + i * threadNum] =
                reinterpret_cast<int64_t*>(aptr)[mmaIdx + (mmaIdy + i * 64) * K / 4];
        }

#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            reinterpret_cast<int64_t*>(smem[1])[thread + i * threadNum] =
                reinterpret_cast<int64_t*>(bptr)[mmbIdx + (mmbIdy + i * 8) * N / 4];
        }
        __syncthreads();

        for (unsigned i = 0; i < WARP_TILE_M; ++i)
        {
            wmma::load_matrix_sync(A_frag[i],
                                   reinterpret_cast<half*>(smem[0]) + (i + m_idx * WARP_TILE_M) * WMMA_K * 16, WMMA_K);
        }

        for (unsigned i = 0; i < WARP_TILE_N; ++i)
        {
            wmma::load_matrix_sync(B_frag[i], reinterpret_cast<half*>(smem[1]) + (i + n_idx * WARP_TILE_N) * 16,
                                   WMMA_N * WARP_TILE);
        }

        // __syncthreads();
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
            for (int j = 0; j < WARP_TILE_N; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < WARP_TILE_M; ++i)
    {
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n + i * WMMA_M * N + j * WMMA_N, C_frag[i][j],
                                    N, wmma::mem_row_major);
        }
    }
}

void hgemm_wmma_m16n16k16_mma8x8(half* A, half* B, half* C, int M, int N, int K)
{
    constexpr int WMMA_M      = 16;
    constexpr int WMMA_N      = 16;
    constexpr int WMMA_K      = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE   = 8;
    dim3          block_dim(32, WMMA_TILE_M, WMMA_TILE_N);
    dim3          grid_dim(div_ceil(N, WMMA_N * WARP_TILE), div_ceil(M, WMMA_M * WARP_TILE));

    hgemm_wmma_m16n16k16_mma8X8_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma8X8_doubleBuf_kernel(half* A, half* B, half* C, int M, int N, int K)
{
    const int     NUM_K_TILES = div_ceil(K, WMMA_K);
    const int     m_idx       = threadIdx.y;
    const int     n_idx       = threadIdx.z;
    const int     thread      = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    constexpr int threadNum   = WMMA_TILE_M * WMMA_TILE_N * warpSize;

    constexpr int WARP_TILE   = 8;
    constexpr int WARP_TILE_M = WARP_TILE / WMMA_TILE_M;
    constexpr int WARP_TILE_N = WARP_TILE / WMMA_TILE_N;

    const int share_mem_a_m = blockIdx.y * WARP_TILE * WMMA_M;
    const int share_mem_b_n = blockIdx.x * WARP_TILE * WMMA_M;

    const int load_gmem_a_m = share_mem_a_m + m_idx * WMMA_M * WARP_TILE_M;
    const int load_gmem_b_n = share_mem_b_n + n_idx * WMMA_N * WARP_TILE_N;
    static_assert(WMMA_N == WMMA_K);

    constexpr int   STAGES = 2;
    __shared__ half smem[STAGES][2][WARP_TILE][WMMA_M * WMMA_N];

    static_assert(WMMA_TILE_M == 4);
    static_assert(WMMA_TILE_N == 2);

    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M][WARP_TILE_N];
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag[WARP_TILE_N];

    unsigned mmaIdx = thread % 4;
    unsigned mmaIdy = thread / 4;
    unsigned mmbIdx = thread % 32;
    unsigned mmbIdy = thread / 32;

    unsigned loadIdx  = 0;
    unsigned storeIdx = 0;

    int kLoadIdx     = 0;
    int kComputerIdx = 0;
    // auto aptr = A + share_mem_a_m * K ;
    // auto bptr = B + share_mem_b_n;
    auto aptr = A + share_mem_a_m * K + kLoadIdx * WMMA_K;
    auto bptr = B + (kLoadIdx * WMMA_K) * N + share_mem_b_n;

    auto loadToSMem = [&](unsigned loadIdx)
    {
#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            reinterpret_cast<int64_t*>(smem[loadIdx][0])[thread + i * threadNum] =
                reinterpret_cast<int64_t*>(aptr)[mmaIdx + (mmaIdy + i * 64) * K / 4];
        }

#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            reinterpret_cast<int64_t*>(smem[loadIdx][1])[thread + i * threadNum] =
                reinterpret_cast<int64_t*>(bptr)[mmbIdx + (mmbIdy + i * 8) * N / 4];
        }
    };

    loadToSMem(loadIdx);
    loadIdx = loadIdx ^ 1;
    kLoadIdx++;

    auto fetchFromSMem = [&](unsigned storeIdx)
    {
        for (unsigned i = 0; i < WARP_TILE_M; ++i)
        {
            wmma::load_matrix_sync(A_frag[i],
                                   reinterpret_cast<half*>(smem[storeIdx][0]) + (i + m_idx * WARP_TILE_M) * WMMA_K * 16,
                                   WMMA_K);
        }

        for (unsigned i = 0; i < WARP_TILE_N; ++i)
        {
            wmma::load_matrix_sync(B_frag[i],
                                   reinterpret_cast<half*>(smem[storeIdx][1]) + (i + n_idx * WARP_TILE_N) * 16,
                                   WMMA_N * WARP_TILE);
        }
    };

    for (; kLoadIdx < NUM_K_TILES; ++kLoadIdx, ++kComputerIdx)
    {
        aptr = A + share_mem_a_m * K + kLoadIdx * WMMA_K;
        bptr = B + (kLoadIdx * WMMA_K) * N + share_mem_b_n;

        __syncthreads();
        loadToSMem(loadIdx);
        loadIdx = loadIdx ^ 1;
        fetchFromSMem(storeIdx);
        storeIdx = storeIdx ^ 1;

        // __syncthreads();
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
            for (int j = 0; j < WARP_TILE_N; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }

    __syncthreads();
    {
        fetchFromSMem(storeIdx);
        storeIdx = storeIdx ^ 1;

        for (int i = 0; i < WARP_TILE_M; ++i)
        {
            for (int j = 0; j < WARP_TILE_N; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        // __syncthreads();
    }

    for (int i = 0; i < WARP_TILE_M; ++i)
    {
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n + i * WMMA_M * N + j * WMMA_N, C_frag[i][j],
                                    N, wmma::mem_row_major);
        }
    }
}

void hgemm_wmma_m16n16k16_mma8x8_doubleBuf(half* A, half* B, half* C, int M, int N, int K)
{
    constexpr int WMMA_M      = 16;
    constexpr int WMMA_N      = 16;
    constexpr int WMMA_K      = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE   = 8;
    dim3          block_dim(32, WMMA_TILE_M, WMMA_TILE_N);
    dim3          grid_dim(div_ceil(N, WMMA_N * WARP_TILE), div_ceil(M, WMMA_M * WARP_TILE));

    hgemm_wmma_m16n16k16_mma8X8_doubleBuf_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

int main()
{
    constexpr unsigned M = 2048;
    constexpr unsigned N = 2048;
    constexpr unsigned K = 2048;
    static_assert(M >= 128 && N >= 128 && K % 16 == 0 && M % 16 == 0 && N % 16 == 0);

    Tester tester(M, N, K, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_mma8x8, "hgemm_wmma_m16n16k16_mma8x8");

    printf(" ======================== \n");
    Tester tester2(M, N, K, 1, 10, 100, true);
    tester2.evaluate(hgemm_wmma_m16n16k16_mma8x8_doubleBuf, "hgemm_wmma_m16n16k16_mma8x8_doubleBuf");

    Matrix* c1 = tester.getC();
    Matrix* c2 = tester2.getC();

    int num = 0;
    for (size_t i = 0; i < c1->getSize(); i++)
    {
        if ((float)c1->getHostPtr()[i] - (float)c2->getHostPtr()[i] > 1e-6)
        {
            num++;
            printf("%d %f %f\n", (int)i, (float)c1->getHostPtr()[i], (float)c2->getHostPtr()[i]);
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