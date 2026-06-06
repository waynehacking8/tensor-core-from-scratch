// Kernel 10: Larger Tiles + Double-Buffered WMMA
//
// Building on kernel 09's double-buffering, we increase the K-tile
// size to BK=32 so each shared memory load feeds TWO rounds of WMMA
// (each WMMA_K=16).  This doubles the compute-to-load ratio.
//
// Combined optimizations:
//   - Double-buffered shared memory (hide memory latency)
//   - BK=32 (2x more compute per load)
//   - 2x2 WMMA tiles per warp (128x128 block tile)
//   - Larger blocks = fewer waves = better SM utilization
//
// This is the highest-performing kernel in the project.
//
// Simplification: C = alpha * A * B (beta=0 only).

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t e = (err);                                                     \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t e = (err);                                                  \
    if (e != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, e);      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int BM = 128;
const int BN = 128;
const int BK = 32; // 2x larger than kernel 09

const int WARPS_M = 4;
const int WARPS_N = 4;
const int NUM_WARPS = WARPS_M * WARPS_N;
const int BLOCK_SIZE = NUM_WARPS * 32;

const int WARP_TILES_M = BM / (WARPS_M * WMMA_M); // 2
const int WARP_TILES_N = BN / (WARPS_N * WMMA_N); // 2
const int K_TILES = BK / WMMA_K;                   // 2

__global__ void hgemm_large_tile(int M, int N, int K, float alpha,
                                 const half *A, const half *B, float *C) {
  __shared__ half As[2][BM][BK + 8];
  __shared__ half Bs[2][BK][BN + 8];

  const int warpId = threadIdx.x / 32;
  const int warpRow = warpId / WARPS_N;
  const int warpCol = warpId % WARPS_N;

  const int blockRowStart = blockIdx.y * BM;
  const int blockColStart = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc[WARP_TILES_M][WARP_TILES_N];
  for (int tm = 0; tm < WARP_TILES_M; tm++)
    for (int tn = 0; tn < WARP_TILES_N; tn++)
      wmma::fill_fragment(acc[tm][tn], 0.0f);

  // Load first tile into buffer 0
  for (int i = threadIdx.x; i < BM * BK; i += BLOCK_SIZE) {
    int r = i / BK, c = i % BK;
    int gr = blockRowStart + r, gc = c;
    As[0][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
  }
  for (int i = threadIdx.x; i < BK * BN; i += BLOCK_SIZE) {
    int r = i / BN, c = i % BN;
    int gr = r, gc = blockColStart + c;
    Bs[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
  }
  __syncthreads();

  int numTiles = (K + BK - 1) / BK;

  for (int tile = 0; tile < numTiles; tile++) {
    int cur = tile % 2;
    int nxt = 1 - cur;
    int nextK = (tile + 1) * BK;

    // Prefetch next tile into the other buffer
    if (tile + 1 < numTiles) {
      for (int i = threadIdx.x; i < BM * BK; i += BLOCK_SIZE) {
        int r = i / BK, c = i % BK;
        int gr = blockRowStart + r, gc = nextK + c;
        As[nxt][r][c] =
            (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
      }
      for (int i = threadIdx.x; i < BK * BN; i += BLOCK_SIZE) {
        int r = i / BN, c = i % BN;
        int gr = nextK + r, gc = blockColStart + c;
        Bs[nxt][r][c] =
            (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
      }
    }

    // Compute: walk K in steps of WMMA_K within the BK tile
    for (int kk = 0; kk < BK; kk += WMMA_K) {
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
          int aRow = (warpRow * WARP_TILES_M + tm) * WMMA_M;
          int bCol = (warpCol * WARP_TILES_N + tn) * WMMA_N;

          wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> b_frag;

          wmma::load_matrix_sync(a_frag, &As[cur][aRow][kk], BK + 8);
          wmma::load_matrix_sync(b_frag, &Bs[cur][kk][bCol], BN + 8);
          wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
        }
      }
    }

    __syncthreads();
  }

  // Store
  for (int tm = 0; tm < WARP_TILES_M; tm++) {
    for (int tn = 0; tn < WARP_TILES_N; tn++) {
      int cRow = blockRowStart + (warpRow * WARP_TILES_M + tm) * WMMA_M;
      int cCol = blockColStart + (warpCol * WARP_TILES_N + tn) * WMMA_N;
      if (cRow < M && cCol < N) {
        for (int i = 0; i < acc[tm][tn].num_elements; i++)
          acc[tm][tn].x[i] *= alpha;
        wmma::store_matrix_sync(C + cRow * N + cCol, acc[tm][tn], N,
                                wmma::mem_row_major);
      }
    }
  }
}

__global__ void fp32_to_fp16(const float *in, half *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

void randomize(float *h, int n) {
  for (int i = 0; i < n; ++i)
    h[i] = static_cast<float>(rand()) / RAND_MAX;
}

float benchmark_cublas(int M, int N, int K, const float *dA, const float *dB,
                       float *dC, int warmup, int iters) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float alpha = 1.0f, beta = 0.0f;
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  return ms / iters;
}

float benchmark_cublas_hgemm(int M, int N, int K, const half *dA,
                             const half *dB, half *dC, int warmup, int iters) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha_h, dB, N, dA, K, &beta_h, dC, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha_h, dB, N, dA, K, &beta_h, dC, N));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  return ms / iters;
}

void verify(const float *ref, const float *test, int n, const char *label) {
  float max_err = 0.0f;
  double sum_err = 0.0;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    sum_err += diff;
    if (diff > max_err) max_err = diff;
  }
  float avg_err = (float)(sum_err / n);
  printf("  %-30s max=%.4f avg=%.6f %s\n", label, max_err, avg_err,
         max_err < 5.0f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Kernel 10: Large Tiles + Double-Buffered WMMA ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

  size_t smem_needed = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
  printf("Double-buffer SMEM: %zu bytes\n\n", smem_needed);

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("Block tile: BM=%d, BN=%d, BK=%d (2x WMMA_K steps per load)\n",
         BM, BN, BK);
  printf("WMMA tiles per warp: %dx%d, K iterations per tile: %d\n",
         WARP_TILES_M, WARP_TILES_N, K_TILES);
  printf("Warps per block: %d\n\n", NUM_WARPS);

  size_t bytes_A = (size_t)M * K * sizeof(float);
  size_t bytes_B = (size_t)K * N * sizeof(float);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  float *hA = (float *)malloc(bytes_A);
  float *hB = (float *)malloc(bytes_B);
  float *hC_ref = (float *)malloc(bytes_C);
  float *hC_test = (float *)malloc(bytes_C);

  srand(42);
  randomize(hA, M * K);
  randomize(hB, K * N);

  float *dA_f32, *dB_f32, *dC;
  half *dA_f16, *dB_f16, *dC_f16;
  CUDA_CHECK(cudaMalloc(&dA_f32, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB_f32, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMalloc(&dA_f16, (size_t)M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB_f16, (size_t)K * N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC_f16, (size_t)M * N * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dA_f32, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB_f32, hB, bytes_B, cudaMemcpyHostToDevice));

  int threads = 256;
  fp32_to_fp16<<<(M * K + threads - 1) / threads, threads>>>(dA_f32, dA_f16, M * K);
  fp32_to_fp16<<<(K * N + threads - 1) / threads, threads>>>(dB_f32, dB_f16, K * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // === cuBLAS FP32 SGEMM reference ===
  float cublas_sgemm_ms = benchmark_cublas(M, N, K, dA_f32, dB_f32, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes_C, cudaMemcpyDeviceToHost));
  double cublas_sgemm_tflops = 2.0 * M * N * K / (cublas_sgemm_ms * 1e-3) * 1e-12;

  // === cuBLAS FP16 HGEMM reference (apples-to-apples) ===
  float cublas_hgemm_ms = benchmark_cublas_hgemm(M, N, K, dA_f16, dB_f16, dC_f16, warmup, iters);
  double cublas_hgemm_tflops = 2.0 * M * N * K / (cublas_hgemm_ms * 1e-3) * 1e-12;

  // === Our kernel ===
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BLOCK_SIZE);

  for (int i = 0; i < warmup; ++i)
    hgemm_large_tile<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hgemm_large_tile<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  kernel_ms /= iters;

  CUDA_CHECK(cudaMemcpy(hC_test, dC, bytes_C, cudaMemcpyDeviceToHost));
  double kernel_tflops = 2.0 * M * N * K / (kernel_ms * 1e-3) * 1e-12;

  printf("%-24s %8.2f ms  %7.2f TFLOPS\n",
         "cuBLAS SGEMM (FP32):", cublas_sgemm_ms, cublas_sgemm_tflops);
  printf("%-24s %8.2f ms  %7.2f TFLOPS\n",
         "cuBLAS HGEMM (FP16):", cublas_hgemm_ms, cublas_hgemm_tflops);
  printf("%-24s %8.2f ms  %7.2f TFLOPS  (%5.1f%% of SGEMM, %5.1f%% of HGEMM)\n",
         "10_large_tile:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / cublas_sgemm_tflops,
         100.0 * kernel_tflops / cublas_hgemm_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "10_large_tile vs cuBLAS FP32");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA_f32)); CUDA_CHECK(cudaFree(dB_f32));
  CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dC_f16));
  CUDA_CHECK(cudaFree(dA_f16)); CUDA_CHECK(cudaFree(dB_f16));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
