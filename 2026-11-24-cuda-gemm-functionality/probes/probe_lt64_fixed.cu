// Activate cuBLASLt fixed-point FP64 emulation via an explicit emulation
// descriptor (strategy=EAGER, mantissa control=FIXED, max bits=N).
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static const int M = 512, N = 512, K = 512;
static void* g_ws = nullptr;
static const size_t g_ws_size = 64ull << 20;

int main() {
    cudaSetDevice(0);
    cudaMalloc(&g_ws, g_ws_size);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    std::vector<double> hA(M*K), hB(K*N);
    unsigned s = 1;
    for (auto& v : hA) { s = s*1664525u+1013904223u; v = (double)((int)(s>>8)%2001-1000)/1000.0; }
    for (auto& v : hB) { s = s*1664525u+1013904223u; v = (double)((int)(s>>8)%2001-1000)/1000.0; }
    void *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, (size_t)M*K*8); cudaMalloc(&dB, (size_t)K*N*8);
    cudaMalloc(&dC, (size_t)M*N*8); cudaMalloc(&dD, (size_t)M*N*8);
    cudaMemcpy(dA, hA.data(), (size_t)M*K*8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), (size_t)K*N*8, cudaMemcpyHostToDevice);

    auto run = [&](cublasComputeType_t ct, int strategy, int mantissa_bits, std::vector<double>& out) {
        cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, ct, CUDA_R_64F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEmulationDesc_t emu = nullptr;
        if (strategy >= 0) {
            cublasLtEmulationDescCreate(&emu);
            int32_t st = strategy;
            cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_STRATEGY, &st, sizeof(st));
            if (mantissa_bits > 0) {
                int32_t mc = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
                int32_t bits = mantissa_bits;
                cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &mc, sizeof(mc));
                cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &bits, sizeof(bits));
            }
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &emu, sizeof(emu));
        }
        cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
        cublasLtMatrixLayoutCreate(&Ad, CUDA_R_64F, M, K, M);
        cublasLtMatrixLayoutCreate(&Bd, CUDA_R_64F, K, N, K);
        cublasLtMatrixLayoutCreate(&Cd, CUDA_R_64F, M, N, M);
        cublasLtMatrixLayoutCreate(&Dd, CUDA_R_64F, M, N, M);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size));
        cublasLtMatmulHeuristicResult_t heur; int nres = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &nres);
        int32_t cap = -1; size_t csz = 0;
        if (st == CUBLAS_STATUS_SUCCESS && nres > 0)
            cublasLtMatmulAlgoCapGetAttribute(&heur.algo, CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT, &cap, sizeof(cap), &csz);
        double a = 1, b = 0;
        cublasStatus_t rs = CUBLAS_STATUS_NOT_SUPPORTED;
        if (st == CUBLAS_STATUS_SUCCESS && nres > 0)
            rs = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd, &heur.algo, g_ws, g_ws_size, 0);
        cudaDeviceSynchronize();
        out.resize((size_t)M*N);
        cudaMemcpy(out.data(), dD, out.size()*8, cudaMemcpyDeviceToHost);
        printf("ct=%d strategy=%d bits=%2d: heuristic=%d n=%d chosen_emu_cap=%d run=%d\n",
               (int)ct, strategy, mantissa_bits, (int)st, nres, cap, (int)rs);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd);
        cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
        if (emu) cublasLtEmulationDescDestroy(emu);
        cublasLtMatmulDescDestroy(op);
    };

    std::vector<double> native, fixed40, fixed55;
    run(CUBLAS_COMPUTE_64F, -1, 0, native);
    run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_EMULATION_STRATEGY_EAGER, 40, fixed40);
    run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_EMULATION_STRATEGY_EAGER, 55, fixed55);

    auto diff = [&](const std::vector<double>& x, const char* name) {
        size_t same = 0; double num = 0, den = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (memcmp(&x[i], &native[i], 8) == 0) ++same;
            num += (x[i]-native[i])*(x[i]-native[i]);
            den += native[i]*native[i];
        }
        printf("%-28s identical=%zu/%zu rel.diff vs native=%.2e\n", name, same, x.size(), std::sqrt(num/den));
    };
    diff(fixed40, "fixed-point 40 bits");
    diff(fixed55, "fixed-point 55 bits");
    return 0;
}
