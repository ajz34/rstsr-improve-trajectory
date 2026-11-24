// Which cross-precision A/B dtype pairs do cuBLAS and cuBLASLt accept?
// Does cuBLAS accept f32 x f64 -> f64 (A and B of different dtypes)?
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static const int M = 256, N = 256, K = 256;

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    std::vector<float> hA32(M*K, 0.5f), hA64(M*K, 0.5);
    std::vector<double> hB64(K*N, 0.25), hB32(K*N, 0.25);
    void *dA32, *dA64, *dB64, *dB32, *dC64;
    cudaMalloc(&dA32, M*K*4); cudaMalloc(&dA64, M*K*8);
    cudaMalloc(&dB64, K*N*8); cudaMalloc(&dB32, K*N*4); cudaMalloc(&dC64, M*N*8);
    cudaMemcpy(dA32, hA32.data(), M*K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dA64, hA64.data(), M*K*8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB64, hB64.data(), K*N*8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB32, hB32.data(), K*N*4, cudaMemcpyHostToDevice);

    double one = 1, zero = 0;
    void *dA16, *dB16, *dC32, *dC16;
    cudaMalloc(&dA16, M*K*2); cudaMalloc(&dB16, K*N*2);
    cudaMalloc(&dC32, M*N*4); cudaMalloc(&dC16, M*N*2);
    cudaMemset(dA16, 0x3c, M*K*2); cudaMemset(dB16, 0x34, K*N*2);
    const char* names[] = {"A=f32 B=f64 -> f64", "A=f64 B=f32 -> f64",
                           "A=f16 B=f32 -> f32", "A=f32 B=f16 -> f32", "A=bf16 B=f32 -> f32"};
    cudaDataType_t ats[] = {CUDA_R_32F, CUDA_R_64F, CUDA_R_16F, CUDA_R_32F, CUDA_R_16BF};
    cudaDataType_t bts[] = {CUDA_R_64F, CUDA_R_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_32F};
    void* aps[] = {dA32, dA64, dA16, dA32, dA16};
    void* bps[] = {dB64, dB32, dB32, dB16, dB32};
    cudaDataType_t cts[] = {CUDA_R_64F, CUDA_R_64F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F};
    void* cps[] = {dC64, dC64, dC32, dC32, dC32};
    cublasComputeType_t comps[] = {CUBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F,
                                   CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F};
    for (int i = 0; i < 5; ++i) {
        cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one,
            aps[i], ats[i], M, bps[i], bts[i], K, &zero, dC64, CUDA_R_64F, M,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        printf("GemmEx   %-20s -> %d\n", names[i], (int)s);
    }
    auto lt_run = [&](int i) {
        cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, comps[i], cts[i]);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayout_t Ad, Bd, Cd;
        cublasLtMatrixLayoutCreate(&Ad, ats[i], M, K, M);
        cublasLtMatrixLayoutCreate(&Bd, bts[i], K, N, K);
        cublasLtMatrixLayoutCreate(&Cd, cts[i], M, N, M);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulHeuristicResult_t heur; int n = 0;
        cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &n);
        if (s == CUBLAS_STATUS_SUCCESS && n > 0) {
            float onef = 1, zerof = 0;
            void* alpha = cts[i] == CUDA_R_64F ? (void*)&one : (void*)&onef;
            void* beta  = cts[i] == CUDA_R_64F ? (void*)&zero : (void*)&zerof;
            s = cublasLtMatmul(lt, op, alpha, aps[i], Ad, bps[i], Bd, beta, cps[i], Cd, cps[i], Cd,
                               &heur.algo, nullptr, 0, 0);
        }
        cudaDeviceSynchronize();
        printf("cuBLASLt %-20s -> %d\n", names[i], (int)s);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatmulDescDestroy(op);
    };
    for (int i = 0; i < 5; ++i) lt_run(i);
    return 0;
}
