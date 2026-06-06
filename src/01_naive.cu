// Kernel 01: Naive Matrix Multiplication
//
// Each thread computes one element of C = A * B by walking the entire K
// dimension.  This is the simplest possible GPU matmul — and the slowest.
// It serves as our baseline: every subsequent kernel is measured against
// this and against cuBLAS.

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------
// Benchmark harness (reused across all kernels)
// ---------------------------------------------------------------------------
void randomize(float *h, int n) {
  for (int i = 0; i < n; ++i)
    h[i] = static_cast<float>(rand()) / RAND_MAX;
}

float benchmark_cublas(int M, int N, int K, const float *dA, const float *dB,
                       float *dC, int warmup, int iters) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float alpha = 1.0f, beta = 0.0f;

  // cuBLAS uses column-major, so we compute B^T * A^T = (AB)^T
  // which gives us C in row-major when we pass: (N, M, K, B, A, C)
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
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    if (diff > max_err)
      max_err = diff;
  }
  printf("  %-30s max error = %.6f %s\n", label, max_err,
         max_err < 1e-2f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Kernel 01: Naive Matrix Multiplication ===\n\n");

  // Print GPU info
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("SMs: %d, Peak TFLOPS (FP32): %.1f\n\n", prop.multiProcessorCount,
         2.0f * prop.clockRate * 1e-6f * prop.multiProcessorCount * 128);

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n\n",
         2.0 * M * N * K * 1e-9);

  // Allocate host memory
  size_t bytes_A = (size_t)M * K * sizeof(float);
  size_t bytes_B = (size_t)K * N * sizeof(float);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  float *hA = (float *)malloc(bytes_A);
  float *hB = (float *)malloc(bytes_B);
  float *hC_cublas = (float *)malloc(bytes_C);
  float *hC_naive = (float *)malloc(bytes_C);

  srand(42);
  randomize(hA, M * K);
  randomize(hB, K * N);

  // Allocate device memory
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));

  CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

  // --- cuBLAS reference ---
  float cublas_ms =
      benchmark_cublas(M, N, K, dA, dB, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_cublas, dC, bytes_C, cudaMemcpyDeviceToHost));

  double cublas_tflops = 2.0 * M * N * K / (cublas_ms * 1e-3) * 1e-12;

  // --- Naive kernel ---
  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  for (int i = 0; i < warmup; ++i)
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i)
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float naive_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
  naive_ms /= iters;

  CUDA_CHECK(cudaMemcpy(hC_naive, dC, bytes_C, cudaMemcpyDeviceToHost));

  double naive_tflops = 2.0 * M * N * K / (naive_ms * 1e-3) * 1e-12;

  // --- Results ---
  printf("%-20s %8.2f ms  %7.2f TFLOPS\n", "cuBLAS:", cublas_ms,
         cublas_tflops);
  printf("%-20s %8.2f ms  %7.2f TFLOPS  (%5.1f%% of cuBLAS)\n",
         "01_naive:", naive_ms, naive_tflops,
         100.0 * naive_tflops / cublas_tflops);
  printf("\n");

  // --- Correctness ---
  verify(hC_cublas, hC_naive, M * N, "01_naive vs cuBLAS");

  // Cleanup
  free(hA);
  free(hB);
  free(hC_cublas);
  free(hC_naive);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
