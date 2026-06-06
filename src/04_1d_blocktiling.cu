// Kernel 04: 1D Block Tiling
//
// In kernel 03, each thread computes ONE output element.  This means the
// ratio of computation to memory access is poor — each loaded value is
// used only once per thread.
//
// Here, each thread computes TM output elements in a column.  We use a
// block of BM×BK threads to load A tiles and BK×BN threads to load B
// tiles into shared memory.  Each thread then accumulates TM results
// using the shared B row it needs.
//
// This increases arithmetic intensity: each B element loaded into a
// register is reused TM times, and each A element loaded from shared
// memory is reused across the inner K loop.

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;

__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int totalThreads = BM * BN / TM;
  const int threadId = threadIdx.x;

  const int threadRow = threadId / (BN);
  const int threadCol = threadId % (BN);

  float results[TM] = {0.0f};

  const float *a_base = A + by * BM * K;
  const float *b_base = B + bx * BN;

  // Indices for loading A and B into shared memory
  const int a_innerRow = threadId / BK;
  const int a_innerCol = threadId % BK;
  const int a_stride = totalThreads / BK;

  const int b_innerRow = threadId / BN;
  const int b_innerCol = threadId % BN;
  const int b_stride = totalThreads / BN;

  for (int bk = 0; bk < K; bk += BK) {
    // Load A tile
    for (int offset = 0; offset < BM; offset += a_stride) {
      int row = a_innerRow + offset;
      if (row < BM) {
        int globalRow = by * BM + row;
        int globalCol = bk + a_innerCol;
        As[row][a_innerCol] = (globalRow < M && globalCol < K)
                                  ? a_base[row * K + bk + a_innerCol]
                                  : 0.0f;
      }
    }

    // Load B tile
    for (int offset = 0; offset < BK; offset += b_stride) {
      int row = b_innerRow + offset;
      if (row < BK) {
        int globalRow = bk + row;
        int globalCol = bx * BN + b_innerCol;
        Bs[row][b_innerCol] = (globalRow < K && globalCol < N)
                                  ? b_base[(bk + row) * N + b_innerCol]
                                  : 0.0f;
      }
    }

    __syncthreads();

    // Compute
    for (int k = 0; k < BK; ++k) {
      float b_val = Bs[k][threadCol];
      for (int tm = 0; tm < TM; ++tm) {
        results[tm] += As[threadRow * TM + tm][k] * b_val;
      }
    }

    __syncthreads();
  }

  // Write results
  for (int tm = 0; tm < TM; ++tm) {
    int globalRow = by * BM + threadRow * TM + tm;
    int globalCol = bx * BN + threadCol;
    if (globalRow < M && globalCol < N) {
      C[globalRow * N + globalCol] =
          alpha * results[tm] + beta * C[globalRow * N + globalCol];
    }
  }
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
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    if (diff > max_err)
      max_err = diff;
  }
  printf("  %-30s max error = %.6f %s\n", label, max_err,
         max_err < 1e-2f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Kernel 04: 1D Block Tiling ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("Block tile: BM=%d, BN=%d, BK=%d, TM=%d\n", BM, BN, BK, TM);
  printf("Threads per block: %d\n\n", BM * BN / TM);

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

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

  float cublas_ms = benchmark_cublas(M, N, K, dA, dB, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes_C, cudaMemcpyDeviceToHost));
  double cublas_tflops = 2.0 * M * N * K / (cublas_ms * 1e-3) * 1e-12;

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BM * BN / TM);

  for (int i = 0; i < warmup; ++i)
    sgemm_1d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    sgemm_1d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  kernel_ms /= iters;

  CUDA_CHECK(cudaMemcpy(hC_test, dC, bytes_C, cudaMemcpyDeviceToHost));
  double kernel_tflops = 2.0 * M * N * K / (kernel_ms * 1e-3) * 1e-12;

  printf("%-20s %8.2f ms  %7.2f TFLOPS\n", "cuBLAS:", cublas_ms,
         cublas_tflops);
  printf("%-20s %8.2f ms  %7.2f TFLOPS  (%5.1f%% of cuBLAS)\n",
         "04_1d_tile:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / cublas_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "04_1d_tile vs cuBLAS");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
