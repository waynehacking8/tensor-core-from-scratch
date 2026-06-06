// Kernel 05: 2D Block Tiling (Register Tiling)
//
// In kernel 04, each thread computes TM elements in one column.  Here,
// each thread computes a TM×TN sub-tile stored entirely in registers.
// This is the key optimization that gets CUDA-core matmul to 50-60% of
// cuBLAS — each shared memory value loaded into a register is reused
// across both the M and N dimensions.
//
// Register file usage: each thread holds TM*TN accumulators + TM values
// from A_shared + TN values from B_shared = TM*TN + TM + TN registers.
// With TM=TN=8, that is 80 registers per thread.

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

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_2d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  const int numThreads = (BM / TM) * (BN / TN);

  // Indices for cooperative loading of A and B tiles
  const int a_innerRow = threadIdx.x / BK;
  const int a_innerCol = threadIdx.x % BK;
  const int a_stride = numThreads / BK;

  const int b_innerRow = threadIdx.x / BN;
  const int b_innerCol = threadIdx.x % BN;
  const int b_stride = numThreads / BN;

  float results[TM][TN] = {{0.0f}};
  float regA[TM];
  float regB[TN];

  for (int bk = 0; bk < K; bk += BK) {
    // Load A tile into shared memory
    for (int offset = 0; offset < BM; offset += a_stride) {
      int row = a_innerRow + offset;
      int globalRow = by * BM + row;
      int globalCol = bk + a_innerCol;
      As[row][a_innerCol] =
          (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
    }

    // Load B tile into shared memory
    for (int offset = 0; offset < BK; offset += b_stride) {
      int row = b_innerRow + offset;
      int globalRow = bk + row;
      int globalCol = bx * BN + b_innerCol;
      Bs[row][b_innerCol] =
          (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
    }

    __syncthreads();

    // Compute TM×TN outer product for each k
    for (int k = 0; k < BK; ++k) {
      for (int tm = 0; tm < TM; ++tm)
        regA[tm] = As[threadRow * TM + tm][k];

      for (int tn = 0; tn < TN; ++tn)
        regB[tn] = Bs[k][threadCol * TN + tn];

      for (int tm = 0; tm < TM; ++tm)
        for (int tn = 0; tn < TN; ++tn)
          results[tm][tn] += regA[tm] * regB[tn];
    }

    __syncthreads();
  }

  // Write results
  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      int globalRow = by * BM + threadRow * TM + tm;
      int globalCol = bx * BN + threadCol * TN + tn;
      if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] =
            alpha * results[tm][tn] + beta * C[globalRow * N + globalCol];
      }
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
  printf("=== Kernel 05: 2D Block Tiling (Register Tiling) ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("Block tile: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
  printf("Threads per block: %d\n", (BM / TM) * (BN / TN));
  printf("Registers per thread (accumulators): %d\n\n", TM * TN);

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
  dim3 block((BM / TM) * (BN / TN));

  for (int i = 0; i < warmup; ++i)
    sgemm_2d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    sgemm_2d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
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
         "05_2d_tile:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / cublas_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "05_2d_tile vs cuBLAS");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
