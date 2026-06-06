// Kernel 08: Raw PTX mma.sync — Direct Tensor Core Control
//
// WMMA hides the register layout from you.  Here we use inline PTX to
// issue mma.sync instructions directly, giving us full control over
// how data flows between shared memory, registers, and tensor cores.
//
// We use mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 which
// computes a 16×8 output from a 16×8 A fragment and an 8×8 B fragment.
// Two of these (along N) give us a 16×16 output per warp.
// Two of these (along K=16) cover a full BK=16 step.
//
// The key insight this kernel teaches is the REGISTER LAYOUT:
//
//   For A (16×8, row-major), each thread holds 4 uint32 registers:
//     groupID = lane_id / 4   (0-7, maps to row pairs)
//     tid     = lane_id % 4   (0-3, maps to column pairs)
//     a[0] = {A[groupID,   tid*2], A[groupID,   tid*2+1]}
//     a[1] = {A[groupID+8, tid*2], A[groupID+8, tid*2+1]}
//
//   For B (8×8, col-major → transposed from row-major B):
//     b[0] = {B[tid*2,   groupID], B[tid*2+1, groupID]}
//
//   For D (16×8 output), each thread holds 4 float registers:
//     d[0] = C[groupID,   tid*2]
//     d[1] = C[groupID,   tid*2+1]
//     d[2] = C[groupID+8, tid*2]
//     d[3] = C[groupID+8, tid*2+1]
//
// Simplification: this kernel computes C = alpha * A * B (beta=0 only).
// A production GEMM would load existing C values for the beta term.

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
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
const int BK = 16;

const int MMA_M = 16;
const int MMA_N = 8;
const int MMA_K = 8;

const int WARP_M = 16;
const int WARP_N = 16; // two m16n8 tiles side by side

const int WARPS_M = BM / WARP_M; // 4
const int WARPS_N = BN / WARP_N; // 4
const int NUM_WARPS = WARPS_M * WARPS_N;
const int BLOCK_SIZE = NUM_WARPS * 32;

// Pack two halfs into a uint32
__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
  uint32_t result;
  asm("mov.b32 %0, {%1, %2};"
      : "=r"(result)
      : "h"(__half_as_ushort(a)), "h"(__half_as_ushort(b)));
  return result;
}

__global__ void hgemm_mma_ptx(int M, int N, int K, float alpha, const half *A,
                              const half *B, float beta, float *C) {
  __shared__ half As[BM][BK + 8]; // +8 padding to avoid bank conflicts
  __shared__ half Bs[BK][BN + 8];

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  const int warpRow = warpId / WARPS_N;
  const int warpCol = warpId % WARPS_N;

  const int blockRowStart = blockIdx.y * BM;
  const int blockColStart = blockIdx.x * BN;

  const int groupID = laneId / 4; // 0-7
  const int tid = laneId % 4;     // 0-3

  // Accumulators for two m16n8 output tiles (left and right)
  float d_left[4] = {0, 0, 0, 0};
  float d_right[4] = {0, 0, 0, 0};

  for (int bk = 0; bk < K; bk += BK) {
    // Cooperative load
    for (int i = threadIdx.x; i < BM * BK; i += BLOCK_SIZE) {
      int r = i / BK, c = i % BK;
      int gr = blockRowStart + r, gc = bk + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < BK * BN; i += BLOCK_SIZE) {
      int r = i / BN, c = i % BN;
      int gr = bk + r, gc = blockColStart + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
    }
    __syncthreads();

    int aBaseRow = warpRow * WARP_M;
    int bBaseCol = warpCol * WARP_N;

    // Walk K dimension in steps of MMA_K=8
    for (int kk = 0; kk < BK; kk += MMA_K) {

      // Load A fragment: 16×8 sub-matrix from As[aBaseRow..+15][kk..kk+7]
      // Each thread loads 2 pairs of halfs (4 uint32 regs)
      uint32_t a_frag[4];
      a_frag[0] = pack_half2(As[aBaseRow + groupID][kk + tid * 2],
                             As[aBaseRow + groupID][kk + tid * 2 + 1]);
      a_frag[1] = pack_half2(As[aBaseRow + groupID + 8][kk + tid * 2],
                             As[aBaseRow + groupID + 8][kk + tid * 2 + 1]);
      // Registers 2,3 are unused for m16n8k8 (would be used by m16n8k16)
      // We pass only a_frag[0..1] to the first mma call, a_frag[2..3] unused
      a_frag[2] = a_frag[0]; // replicate — some implementations need this
      a_frag[3] = a_frag[1];

      // Load B fragment for LEFT tile: 8×8 from Bs[kk..kk+7][bBaseCol..+7]
      // B is row-major in shared but mma wants col-major.
      // For m16n8k8 col-major B: b[0] = {B[tid*2, groupID], B[tid*2+1, groupID]}
      // But groupID ranges 0-7 which is the N dimension (only 8 wide) ✓
      uint32_t b_left[2];
      b_left[0] = pack_half2(Bs[kk + tid * 2][bBaseCol + groupID],
                             Bs[kk + tid * 2 + 1][bBaseCol + groupID]);
      // Second half not needed for m16n8k8 with k=8
      b_left[1] = b_left[0]; // padding

      // Load B fragment for RIGHT tile: 8×8 from Bs[kk..+7][bBaseCol+8..+15]
      uint32_t b_right[2];
      b_right[0] = pack_half2(Bs[kk + tid * 2][bBaseCol + 8 + groupID],
                              Bs[kk + tid * 2 + 1][bBaseCol + 8 + groupID]);
      b_right[1] = b_right[0];

      // Issue mma.sync for left output tile (m16n8k8)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5}, "
          "{%6}, "
          "{%7, %8, %9, %10};\n"
          : "=f"(d_left[0]), "=f"(d_left[1]), "=f"(d_left[2]),
            "=f"(d_left[3])
          : "r"(a_frag[0]), "r"(a_frag[1]),
            "r"(b_left[0]),
            "f"(d_left[0]), "f"(d_left[1]), "f"(d_left[2]),
            "f"(d_left[3]));

      // Issue mma.sync for right output tile (reuse A fragment)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5}, "
          "{%6}, "
          "{%7, %8, %9, %10};\n"
          : "=f"(d_right[0]), "=f"(d_right[1]), "=f"(d_right[2]),
            "=f"(d_right[3])
          : "r"(a_frag[0]), "r"(a_frag[1]),
            "r"(b_right[0]),
            "f"(d_right[0]), "f"(d_right[1]), "f"(d_right[2]),
            "f"(d_right[3]));
    }

    __syncthreads();
  }

  // Store results
  // m16n8k8 output layout:
  //   d[0] = C[groupID,   tid*2]
  //   d[1] = C[groupID,   tid*2+1]
  //   d[2] = C[groupID+8, tid*2]
  //   d[3] = C[groupID+8, tid*2+1]

  int outRowBase = blockRowStart + warpRow * WARP_M;

  // Left tile (N offset = 0..7)
  {
    int outColBase = blockColStart + warpCol * WARP_N;
    int r0 = outRowBase + groupID;
    int r1 = outRowBase + groupID + 8;
    int c0 = outColBase + tid * 2;
    int c1 = outColBase + tid * 2 + 1;

    if (r0 < M && c0 < N) C[r0 * N + c0] = alpha * d_left[0];
    if (r0 < M && c1 < N) C[r0 * N + c1] = alpha * d_left[1];
    if (r1 < M && c0 < N) C[r1 * N + c0] = alpha * d_left[2];
    if (r1 < M && c1 < N) C[r1 * N + c1] = alpha * d_left[3];
  }

  // Right tile (N offset = 8..15)
  {
    int outColBase = blockColStart + warpCol * WARP_N + MMA_N;
    int r0 = outRowBase + groupID;
    int r1 = outRowBase + groupID + 8;
    int c0 = outColBase + tid * 2;
    int c1 = outColBase + tid * 2 + 1;

    if (r0 < M && c0 < N) C[r0 * N + c0] = alpha * d_right[0];
    if (r0 < M && c1 < N) C[r0 * N + c1] = alpha * d_right[1];
    if (r1 < M && c0 < N) C[r1 * N + c0] = alpha * d_right[2];
    if (r1 < M && c1 < N) C[r1 * N + c1] = alpha * d_right[3];
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
  printf("=== Kernel 08: Raw PTX mma.sync.m16n8k8 ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 5, iters = 20;

  printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("FLOPs per matmul: %.2f GFLOP\n", 2.0 * M * N * K * 1e-9);
  printf("MMA shape: m%dn%dk%d\n", MMA_M, MMA_N, MMA_K);
  printf("Warp tile: %d×%d (2 mma tiles along N)\n", WARP_M, WARP_N);
  printf("Block tile: %d×%d×%d\n", BM, BN, BK);
  printf("Warps per block: %d (%d×%d)\n\n", NUM_WARPS, WARPS_M, WARPS_N);

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
  half *dA_f16, *dB_f16;
  CUDA_CHECK(cudaMalloc(&dA_f32, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB_f32, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMalloc(&dA_f16, (size_t)M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB_f16, (size_t)K * N * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dA_f32, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB_f32, hB, bytes_B, cudaMemcpyHostToDevice));

  int threads = 256;
  fp32_to_fp16<<<(M * K + threads - 1) / threads, threads>>>(dA_f32, dA_f16, M * K);
  fp32_to_fp16<<<(K * N + threads - 1) / threads, threads>>>(dB_f32, dB_f16, K * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float cublas_ms = benchmark_cublas(M, N, K, dA_f32, dB_f32, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes_C, cudaMemcpyDeviceToHost));
  double cublas_tflops = 2.0 * M * N * K / (cublas_ms * 1e-3) * 1e-12;

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BLOCK_SIZE);

  for (int i = 0; i < warmup; ++i)
    hgemm_mma_ptx<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hgemm_mma_ptx<<<grid, block>>>(M, N, K, alpha, dA_f16, dB_f16, beta, dC);
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
         "08_mma_ptx:", kernel_ms, kernel_tflops,
         100.0 * kernel_tflops / cublas_tflops);
  printf("\n");

  verify(hC_ref, hC_test, M * N, "08_mma_ptx vs cuBLAS FP32");

  free(hA); free(hB); free(hC_ref); free(hC_test);
  CUDA_CHECK(cudaFree(dA_f32)); CUDA_CHECK(cudaFree(dB_f32));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dA_f16)); CUDA_CHECK(cudaFree(dB_f16));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
