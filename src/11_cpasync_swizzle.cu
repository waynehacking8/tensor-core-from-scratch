// Kernel 11: cp.async 3-Stage Pipeline
//
// This kernel uses cp.async (asynchronous global→shared memory copy)
// with a 3-stage shared memory pipeline.  cp.async bypasses registers:
// data goes directly from L2/DRAM to shared memory via a dedicated
// hardware path, freeing registers for computation.
//
// The 3-stage pipeline works like a ring buffer:
//   - PREFILL: issue async copies for stages 0, 1, 2
//   - MAIN LOOP (each iteration):
//     1. __pipeline_wait_prior(2) — wait for the oldest stage
//     2. __syncthreads() — all threads see the data
//     3. COMPUTE on the current stage
//     4. __syncthreads() — compute done, stage is free
//     5. PREFETCH next tile into the freed stage via cp.async
//     6. __pipeline_commit() — ALWAYS commit (advances pipeline counter)
//     7. Advance stage = (stage + 1) % 3
//
// Reference: NVIDIA CUDA Programming Guide §4.11 (Asynchronous Data Copies)
// Simplification: C = alpha * A * B (beta=0 only).

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
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

const int BM = 192;
const int BN = 256;
const int BK = 48;
const int STAGES = 2;

const int WARPS_M = 2;
const int WARPS_N = 4;
const int NUM_WARPS = WARPS_M * WARPS_N;
const int BLOCK_SIZE = NUM_WARPS * 32;

const int WARP_TILES_M = BM / (WARPS_M * WMMA_M); // 2
const int WARP_TILES_N = BN / (WARPS_N * WMMA_N); // 2

const int A_SMEM_STRIDE = BK + 8;
const int B_SMEM_STRIDE = BN + 8;
const int STAGE_A_SIZE = BM * A_SMEM_STRIDE;
const int STAGE_B_SIZE = BK * B_SMEM_STRIDE;
const int STAGE_SIZE = STAGE_A_SIZE + STAGE_B_SIZE;

__device__ __forceinline__ void async_copy_16B(void *dst, const void *src,
                                                bool valid) {
  uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  int src_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
               ::"r"(dst_addr), "l"(src), "n"(16), "r"(src_bytes));
}

__device__ void load_stage_async(half *smem, int stage, const half *A,
                                  const half *B, int M, int N, int K, int bk,
                                  int blockRowStart, int blockColStart) {
  half *As = smem + stage * STAGE_SIZE;
  half *Bs = smem + stage * STAGE_SIZE + STAGE_A_SIZE;

  // Load A[BM][BK]: 128 rows × 32 cols = 4096 halfs, each thread copies 8 halfs
  for (int i = threadIdx.x; i < BM * (BK / 8); i += blockDim.x) {
    int row = i / (BK / 8);
    int col8 = (i % (BK / 8)) * 8;
    int gRow = blockRowStart + row;
    int gCol = bk + col8;
    bool valid = (gRow < M) && (gCol + 7 < K);
    async_copy_16B(As + row * A_SMEM_STRIDE + col8,
                   A + gRow * K + gCol, valid);
  }

  // Load B[BK][BN]: 32 rows × 128 cols = 4096 halfs
  for (int i = threadIdx.x; i < BK * (BN / 8); i += blockDim.x) {
    int row = i / (BN / 8);
    int col8 = (i % (BN / 8)) * 8;
    int gRow = bk + row;
    int gCol = blockColStart + col8;
    bool valid = (gRow < K) && (gCol + 7 < N);
    async_copy_16B(Bs + row * B_SMEM_STRIDE + col8,
                   B + gRow * N + gCol, valid);
  }
}

__global__ void hgemm_cpasync(int M, int N, int K, float alpha,
                               const half *A, const half *B, float *C) {
  extern __shared__ half smem[];

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

  int numKTiles = (K + BK - 1) / BK;

  // === PREFILL: issue async copies for first STAGES tiles ===
  for (int s = 0; s < STAGES && s < numKTiles; s++) {
    load_stage_async(smem, s, A, B, M, N, K, s * BK,
                     blockRowStart, blockColStart);
    __pipeline_commit();
  }

  // === MAIN LOOP ===
  int stage = 0;
  for (int tile = 0; tile < numKTiles; tile++) {
    // Step 1: Wait for the current stage's async copies to complete.
    // wait_prior(STAGES-1) means: wait until at most (STAGES-1)
    // committed batches remain in flight. Since we have STAGES total,
    // the oldest one (our current stage) must be done.
    __pipeline_wait_prior(STAGES - 1);

    // Step 2: All threads must see the completed data
    __syncthreads();

    // Step 3: COMPUTE on the current stage
    half *As_cur = smem + stage * STAGE_SIZE;
    half *Bs_cur = smem + stage * STAGE_SIZE + STAGE_A_SIZE;

    for (int kk = 0; kk < BK; kk += WMMA_K) {
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
          int aRow = (warpRow * WARP_TILES_M + tm) * WMMA_M;
          int bCol = (warpCol * WARP_TILES_N + tn) * WMMA_N;

          wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         wmma::row_major> b_frag;

          wmma::load_matrix_sync(a_frag, As_cur + aRow * A_SMEM_STRIDE + kk,
                                 A_SMEM_STRIDE);
          wmma::load_matrix_sync(b_frag, Bs_cur + kk * B_SMEM_STRIDE + bCol,
                                 B_SMEM_STRIDE);
          wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
        }
      }
    }

    // Step 4: Compute done — this stage's smem is now free
    __syncthreads();

    // Step 5: PREFETCH next tile into the freed stage
    int prefetch_tile = tile + STAGES;
    if (prefetch_tile < numKTiles) {
      load_stage_async(smem, stage, A, B, M, N, K, prefetch_tile * BK,
                       blockRowStart, blockColStart);
    }

    // Step 6: ALWAYS commit to advance the pipeline counter
    __pipeline_commit();

    // Step 7: Advance to next stage
    stage = (stage + 1) % STAGES;
  }

  // Store results
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
                              const float *dB, float *dC, int w, int it) {
  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
  float alpha=1,beta=0;
  for(int i=0;i<w;i++) CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for(int i=0;i<it;i++) CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms; cudaEventElapsedTime(&ms,t0,t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  CUBLAS_CHECK(cublasDestroy(handle)); return ms/it;
}

float benchmark_cublas_hgemm(int M, int N, int K, const half *dA,
                              const half *dB, half *dC, int w, int it) {
  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
  half alpha=__float2half(1),beta=__float2half(0);
  for(int i=0;i<w;i++) CUBLAS_CHECK(cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for(int i=0;i<it;i++) CUBLAS_CHECK(cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms; cudaEventElapsedTime(&ms,t0,t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  CUBLAS_CHECK(cublasDestroy(handle)); return ms/it;
}

void verify(const float *ref, const float *test, int n, const char *label) {
  float mx=0; double sum=0;
  for(int i=0;i<n;i++){float d=fabsf(ref[i]-test[i]);sum+=d;if(d>mx)mx=d;}
  printf("  %-35s max=%.4f avg=%.6f %s\n",label,mx,(float)(sum/n),mx<5.0f?"[PASS]":"[FAIL]");
}

int main() {
  printf("=== Kernel 11: cp.async 3-Stage Pipeline ===\n\n");
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  printf("GPU: %s (SM %d.%d)\n",prop.name,prop.major,prop.minor);

  size_t smem = STAGES * STAGE_SIZE * sizeof(half);
  printf("SMEM: %zu bytes (max %zu)\n\n", smem, prop.sharedMemPerBlockOptin);
  if(smem>prop.sharedMemPerBlockOptin){printf("TOO LARGE\n");return 1;}

  const int M=4096,N=4096,K=4096; const float alpha=1;
  const int warmup=5,iters=20;
  printf("Problem: %dx%dx%d, BM=%d BN=%d BK=%d Stages=%d\n",M,N,K,BM,BN,BK,STAGES);
  printf("Warps: %d, Threads: %d, Warp tiles: %dx%d\n\n",NUM_WARPS,BLOCK_SIZE,WARP_TILES_M,WARP_TILES_N);

  size_t bA=(size_t)M*K*sizeof(float),bB=(size_t)K*N*sizeof(float),bC=(size_t)M*N*sizeof(float);
  float *hA=(float*)malloc(bA),*hB=(float*)malloc(bB),*hCr=(float*)malloc(bC),*hCt=(float*)malloc(bC);
  srand(42); randomize(hA,M*K); randomize(hB,K*N);

  float *dAf,*dBf,*dC; half *dAh,*dBh,*dCh;
  CUDA_CHECK(cudaMalloc(&dAf,bA)); CUDA_CHECK(cudaMalloc(&dBf,bB));
  CUDA_CHECK(cudaMalloc(&dC,bC));
  CUDA_CHECK(cudaMalloc(&dAh,(size_t)M*K*2)); CUDA_CHECK(cudaMalloc(&dBh,(size_t)K*N*2));
  CUDA_CHECK(cudaMalloc(&dCh,(size_t)M*N*2));
  CUDA_CHECK(cudaMemcpy(dAf,hA,bA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBf,hB,bB,cudaMemcpyHostToDevice));

  int t=256;
  fp32_to_fp16<<<(M*K+t-1)/t,t>>>(dAf,dAh,M*K);
  fp32_to_fp16<<<(K*N+t-1)/t,t>>>(dBf,dBh,K*N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float sms=benchmark_cublas_sgemm(M,N,K,dAf,dBf,dC,warmup,iters);
  CUDA_CHECK(cudaMemcpy(hCr,dC,bC,cudaMemcpyDeviceToHost));
  double st=2.0*M*N*K/(sms*1e-3)*1e-12;

  float hms=benchmark_cublas_hgemm(M,N,K,dAh,dBh,dCh,warmup,iters);
  double ht=2.0*M*N*K/(hms*1e-3)*1e-12;

  CUDA_CHECK(cudaFuncSetAttribute(hgemm_cpasync,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  dim3 grid((N+BN-1)/BN,(M+BM-1)/BM), block(BLOCK_SIZE);

  for(int i=0;i<warmup;i++)
    hgemm_cpasync<<<grid,block,smem>>>(M,N,K,alpha,dAh,dBh,dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start);
  for(int i=0;i<iters;i++)
    hgemm_cpasync<<<grid,block,smem>>>(M,N,K,alpha,dAh,dBh,dC);
  cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
  float kms; cudaEventElapsedTime(&kms,ev_start,ev_stop); kms/=iters;

  CUDA_CHECK(cudaMemcpy(hCt,dC,bC,cudaMemcpyDeviceToHost));
  double kt=2.0*M*N*K/(kms*1e-3)*1e-12;

  printf("%-24s %8.2f ms  %7.2f TFLOPS\n","cuBLAS SGEMM (FP32):",sms,st);
  printf("%-24s %8.2f ms  %7.2f TFLOPS\n","cuBLAS HGEMM (FP16):",hms,ht);
  printf("%-24s %8.2f ms  %7.2f TFLOPS  (%5.1f%% SGEMM, %5.1f%% HGEMM)\n",
         "11_cpasync:",kms,kt,100*kt/st,100*kt/ht);
  printf("\n");
  verify(hCr,hCt,M*N,"11_cpasync vs cuBLAS FP32");

  free(hA);free(hB);free(hCr);free(hCt);
  CUDA_CHECK(cudaFree(dAf));CUDA_CHECK(cudaFree(dBf));CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dCh));CUDA_CHECK(cudaFree(dAh));CUDA_CHECK(cudaFree(dBh));
  cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
  return 0;
}
