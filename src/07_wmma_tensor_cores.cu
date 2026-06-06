// Kernel 07: WMMA Tensor Cores — Your First Tensor Core Kernel
//
// This is the transition point: we leave CUDA cores behind and enter the
// world of tensor cores.  NVIDIA's WMMA (Warp Matrix Multiply Accumulate)
// API provides a high-level C++ interface to tensor core operations.
//
// A single WMMA operation computes D = A * B + C where:
//   A is a 16×16 matrix of half (FP16)
//   B is a 16×16 matrix of half (FP16)
//   C/D are 16×16 matrices of float (FP32 accumulation)
//
// The entire warp (32 threads) cooperates on this 16×16×16 operation,
// which delivers 16×16×16×2 = 8192 FLOPs per warp instruction —
// compared to 32×2 = 64 FLOPs for a warp of FMA instructions.
// That is a 128× theoretical throughput advantage.
//
// We tile the problem using WMMA fragments:
//   - Load 16×16 tiles of A and B as FP16
//   - Accumulate in FP32
//   - Store the FP32 result
//
// Note: input data is FP32 and we convert to FP16 on the fly.  In
// production you would keep data in FP16 throughout.  Here we convert
// to keep the benchmark comparable to our FP32 CUDA-core kernels.
//
// Simplification: this kernel computes C = alpha * A * B (beta=0 only).
// A production GEMM would load existing C values for the beta term.

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

// Each warp computes one 16×16 output tile.
// Block: WARP_ROWS × WARP_COLS warps, each handling one 16×16 tile.
const int WARP_ROWS = 4;
const int WARP_COLS = 4;
const int BLOCK_ROWS = WARP_ROWS * WMMA_M; // 64
const int BLOCK_COLS = WARP_COLS * WMMA_N; // 64

__global__ void hgemm_wmma(int M, int N, int K, float alpha, const half *A,
                           const half *B, float beta, float *C) {
  const int warpId = threadIdx.x / 32;
  const int warpRow = warpId / WARP_COLS;
  const int warpCol = warpId % WARP_COLS;

  const int blockRowStart = blockIdx.y * BLOCK_ROWS;
  const int blockColStart = blockIdx.x * BLOCK_COLS;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k = 0; k < K; k += WMMA_K) {
    int aRow = blockRowStart + warpRow * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = blockColStart + warpCol * WMMA_N;

    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                     wmma::row_major>
          b_frag;

      wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
      wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
  }

  int cRow = blockRowStart + warpRow * WMMA_M;
  int cCol = blockColStart + warpCol * WMMA_N;

  if (cRow < M && cCol < N) {
    // Apply alpha scaling
    for (int i = 0; i < acc.num_elements; ++i)
      acc.x[i] *= alpha;

    wmma::store_matrix_sync(C + cRow * N + cCol, acc, N,
                            wmma::mem_row_major);
  }
}

// Convert FP32 array to FP16
__global__ void fp32_to_fp16(const float *in, half *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = __float2half(in[i]);
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

void verify(const float *ref, const float *test, int n, const char *label) {
  float max_err = 0.0f;
  double sum_err = 0.0;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    sum_err += diff;
    if (diff > max_err)
      max_err = diff;
  }
  float avg_err = (float)(sum_err / n);
  // FP16 inputs lose precision; use looser tolerance
  printf("  %-30s max=%.4f avg=%.6f %s\n", label, max_err, avg_err,
         max_err < 5.0f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Kernel 07: WMMA Tensor Cores (FP16 input, FP32 accum) ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Tensor cores available: %s\n\n",
         prop.major >= 7 ? "YES" : "NO");

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("WMMA tile: %d×%d×%d\n", WMMA_M, WMMA_N, WMMA_K);
  printf("Warps per block: %d×%d = %d\n\n", WARP_ROWS, WARP_COLS,
         WARP_ROWS * WARP_COLS);

  size_t bytes_A = (size_t)M * K * sizeof(float);
  size_t bytes_B = (size_t)K * N * sizeof(float);
  size_t bytes_C = (size_t)M * N * sizeof(float);
  size_t half_A = (size_t)M * K * sizeof(half);
  size_t half_B = (size_t)K * N * sizeof(half);

  float *hA = (float *)malloc(bytes_A);
  float *hB = (float *)malloc(bytes_B);
  float *hC_ref = (float *)malloc(bytes_C);
  float *hC_test = (float *)malloc(bytes_C);

  srand(42);
  randomize(hA, M * K);
  randomize(hB, K * N);

  float *dA_f32, *dB_f32, *dC;
  half *dA_f16, *dB_f16;
  CUDA_CHECK(cudaMalloc(&dA_f32, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB_f32, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMalloc(&dA_f16, half_A));
  CUDA_CHECK(cudaMalloc(&dB_f16, half_B));

  CUDA_CHECK(cudaMemcpy(dA_f32, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB_f32, hB, bytes_B, cudaMemcpyHostToDevice));

  // Convert to FP16
  int threads = 256;
  fp32_to_fp16<<<(M * K + threads - 1) / threads, threads>>>(dA_f32, dA_f16,
                                                              M * K);
  fp32_to_fp16<<<(K * N + threads - 1) / threads, threads>>>(dB_f32, dB_f16,
                                                              K * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // cuBLAS reference (FP32)
  float cublas_ms =
      benchmark_cublas(M, N, K, dA_f32, dB_f32, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes_C, cudaMemcpyDeviceToHost));
  double cublas_tflops = 2.0 * M * N * K / (cublas_ms * 1e-3) * 1e-12;

  // WMMA kernel
  dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS,
            (M + BLOCK_ROWS - 1) / BLOCK_ROWS);
  dim3 block(WARP_ROWS * WARP_COLS * 32);

  for (int i = 0; i < warmup; ++i)
    hgemm_wmma<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hgemm_wmma<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, beta, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  kernel_ms /= iters;

  CUDA_CHECK(cudaMemcpy(hC_test, dC, bytes_C, cudaMemcpyDeviceToHost));
  double kernel_tflops = 2.0 * M * N * K / (kernel_ms * 1e-3) * 1e-12;

  printf("%-20s %8.2f ms  %7.2f TFLOPS  (FP32 SGEMM reference)\n",
         "cuBLAS FP32:", cublas_ms, cublas_tflops);
  printf("%-20s %8.2f ms  %7.2f TFLOPS  (%5.1f%% of cuBLAS FP32)\n",
         "07_wmma:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / cublas_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "07_wmma vs cuBLAS FP32");
  printf("  (Higher error expected: FP16 inputs lose precision vs FP32)\n");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA_f32)); CUDA_CHECK(cudaFree(dB_f32));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dA_f16)); CUDA_CHECK(cudaFree(dB_f16));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
