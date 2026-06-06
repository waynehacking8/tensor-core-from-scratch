// Kernel 11: 3-Stage Pipeline + Larger Warp Tiles
//
// Kernels 09-10 use double buffering (2 stages). Here we add a third
// stage so the GPU can have TWO prefetches in flight while computing
// on a third — better hiding of memory latency for high-bandwidth
// workloads.
//
// We also increase the per-warp work: each warp computes 4 WMMA tiles
// (2×2 arrangement) per K-iteration, and we unroll the K-loop within
// each BK tile (BK=32 = 2 × WMMA_K=16).
//
// The combination of 3-stage pipeline + larger warp tiles + BK=32
// pushes compute utilization higher by ensuring warps always have
// data ready to process.
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

const int BM = 256;
const int BN = 128;
const int BK = 32;
const int STAGES = 3;

const int WARPS_M = 8;
const int WARPS_N = 4;
const int NUM_WARPS = WARPS_M * WARPS_N; // 32
const int BLOCK_SIZE = NUM_WARPS * 32;   // 1024

const int WARP_TILES_M = BM / (WARPS_M * WMMA_M); // 2
const int WARP_TILES_N = BN / (WARPS_N * WMMA_N); // 2

__global__ void hgemm_3stage(int M, int N, int K, float alpha,
                              const half *A, const half *B, float *C) {
  const int A_STRIDE = BK + 8;
  const int B_STRIDE = BN + 8;

  extern __shared__ half smem[];
  const int stage_a = BM * A_STRIDE;
  const int stage_b = BK * B_STRIDE;
  const int stage_total = stage_a + stage_b;

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

  auto load_stage = [&](int stage, int bk) {
    half *As = smem + stage * stage_total;
    half *Bs = smem + stage * stage_total + stage_a;

    for (int i = threadIdx.x; i < BM * BK; i += BLOCK_SIZE) {
      int r = i / BK, c = i % BK;
      int gr = blockRowStart + r, gc = bk + c;
      As[r * A_STRIDE + c] =
          (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < BK * BN; i += BLOCK_SIZE) {
      int r = i / BN, c = i % BN;
      int gr = bk + r, gc = blockColStart + c;
      Bs[r * B_STRIDE + c] =
          (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
    }
  };

  int numKTiles = (K + BK - 1) / BK;

  // Prefill: load first min(STAGES, numKTiles) stages
  int filled = 0;
  for (int s = 0; s < STAGES && s < numKTiles; s++) {
    load_stage(s, s * BK);
    filled++;
  }
  __syncthreads();

  for (int tile = 0; tile < numKTiles; tile++) {
    int cur = tile % STAGES;

    // Compute on current stage FIRST (data is ready from prefill or prior prefetch)
    half *As_cur = smem + cur * stage_total;
    half *Bs_cur = smem + cur * stage_total + stage_a;

    for (int kk = 0; kk < BK; kk += WMMA_K) {
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
          int aRow = (warpRow * WARP_TILES_M + tm) * WMMA_M;
          int bCol = (warpCol * WARP_TILES_N + tn) * WMMA_N;

          wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> b_frag;

          wmma::load_matrix_sync(a_frag, As_cur + aRow * A_STRIDE + kk, A_STRIDE);
          wmma::load_matrix_sync(b_frag, Bs_cur + kk * B_STRIDE + bCol, B_STRIDE);
          wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
        }
      }
    }

    __syncthreads();

    // THEN prefetch into the stage we just consumed (safe — compute is done)
    int prefetchTile = tile + STAGES;
    if (prefetchTile < numKTiles) {
      load_stage(cur, prefetchTile * BK);
    }

    __syncthreads();
  }

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

float benchmark_cublas_sgemm(int M, int N, int K, const float *dA,
                              const float *dB, float *dC, int warmup,
                              int iters) {
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
                              const half *dB, half *dC, int warmup,
                              int iters) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
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

void verify(const float *ref, const float *test, int n, const char *label) {
  float max_err = 0.0f;
  double sum_err = 0.0;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    sum_err += diff;
    if (diff > max_err) max_err = diff;
  }
  float avg_err = (float)(sum_err / n);
  printf("  %-35s max=%.4f avg=%.6f %s\n", label, max_err, avg_err,
         max_err < 5.0f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Kernel 11: 3-Stage Pipeline + BM=256 ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Max SMEM (optin): %zu bytes\n", prop.sharedMemPerBlockOptin);

  size_t smem_needed = STAGES * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
  printf("3-stage SMEM: %zu bytes %s\n\n", smem_needed,
         smem_needed <= prop.sharedMemPerBlockOptin ? "(OK)" : "(TOO LARGE)");

  if (smem_needed > prop.sharedMemPerBlockOptin) {
    printf("Need %zu but have %zu — reduce BM or STAGES\n",
           smem_needed, prop.sharedMemPerBlockOptin);
    return 1;
  }

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f;
  const int warmup = 5, iters = 20;

  printf("Problem: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("Block: BM=%d, BN=%d, BK=%d, Stages=%d\n", BM, BN, BK, STAGES);
  printf("Warps: %d (%dx%d), Threads: %d\n", NUM_WARPS, WARPS_M, WARPS_N,
         BLOCK_SIZE);
  printf("Warp tiles: %dx%d\n\n", WARP_TILES_M, WARP_TILES_N);

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

  float sgemm_ms = benchmark_cublas_sgemm(M, N, K, dA_f32, dB_f32, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes_C, cudaMemcpyDeviceToHost));
  double sgemm_tflops = 2.0 * M * N * K / (sgemm_ms * 1e-3) * 1e-12;

  float hgemm_ms = benchmark_cublas_hgemm(M, N, K, dA_f16, dB_f16, dC_f16, warmup, iters);
  double hgemm_tflops = 2.0 * M * N * K / (hgemm_ms * 1e-3) * 1e-12;

  CUDA_CHECK(cudaFuncSetAttribute(hgemm_3stage,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_needed));

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BLOCK_SIZE);

  for (int i = 0; i < warmup; ++i)
    hgemm_3stage<<<grid, block, smem_needed>>>(M, N, K, alpha, dA_f16, dB_f16, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hgemm_3stage<<<grid, block, smem_needed>>>(M, N, K, alpha, dA_f16, dB_f16, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  kernel_ms /= iters;

  CUDA_CHECK(cudaMemcpy(hC_test, dC, bytes_C, cudaMemcpyDeviceToHost));
  double kernel_tflops = 2.0 * M * N * K / (kernel_ms * 1e-3) * 1e-12;

  printf("%-24s %8.2f ms  %7.2f TFLOPS\n", "cuBLAS SGEMM (FP32):", sgemm_ms,
         sgemm_tflops);
  printf("%-24s %8.2f ms  %7.2f TFLOPS\n", "cuBLAS HGEMM (FP16):", hgemm_ms,
         hgemm_tflops);
  printf("%-24s %8.2f ms  %7.2f TFLOPS  (%5.1f%% SGEMM, %5.1f%% HGEMM)\n",
         "11_3stage:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / sgemm_tflops,
         100.0 * kernel_tflops / hgemm_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "11_3stage vs cuBLAS FP32");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA_f32)); CUDA_CHECK(cudaFree(dB_f32));
  CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dC_f16));
  CUDA_CHECK(cudaFree(dA_f16)); CUDA_CHECK(cudaFree(dB_f16));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
