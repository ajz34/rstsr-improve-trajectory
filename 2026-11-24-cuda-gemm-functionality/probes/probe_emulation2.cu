// Second, cleaner attempt: does emulation engage, or fall back to native?
// Separate C and D buffers (Lt may not tolerate aliasing on these paths).
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

static void fill(float* a, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; a[i] = (float)((int)(s >> 8) % 2001 - 1000) / 1000.0f; }
}

// Returns the status; fills D. If emu_strategy >= 0 an emulation descriptor is attached.
static cublasStatus_t run_lt(cublasLtHandle_t lt, cublasComputeType_t ct, cudaDataType_t dt,
                             const void* A, const void* B, void* C, void* D,
                             int strategy, int32_t* mantissa_out) {
    cublasLtMatmulDesc_t op = nullptr;
    cublasStatus_t s = cublasLtMatmulDescCreate(&op, ct, dt == CUDA_R_64F ? CUDA_R_64F : CUDA_R_32F);
    if (s) return s;
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtEmulationDesc_t emu = nullptr;
    int32_t* d_mant = nullptr;
    if (strategy >= 0) {
        s = cublasLtEmulationDescCreate(&emu);
        if (s) { cublasLtMatmulDescDestroy(op); return s; }
        int32_t strat = strategy;
        s = cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_STRATEGY, &strat, sizeof(strat));
        if (s) { cublasLtEmulationDescDestroy(emu); cublasLtMatmulDescDestroy(op); return s; }
        if (mantissa_out) {
            cudaMalloc(&d_mant, 4);
            cudaMemset(d_mant, 0, 4);
            cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,
                                              &d_mant, sizeof(d_mant));
        }
        s = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &emu, sizeof(emu));
        if (s) { cublasLtEmulationDescDestroy(emu); cublasLtMatmulDescDestroy(op); return s; }
    }

    cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
    cublasLtMatrixLayoutCreate(&Ad, dt, M, K, M);
    cublasLtMatrixLayoutCreate(&Bd, dt, K, N, K);
    cublasLtMatrixLayoutCreate(&Cd, dt, M, N, M);
    cublasLtMatrixLayoutCreate(&Dd, dt, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size));
    cublasLtMatmulHeuristicResult_t heur; int nres = 0;
    s = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &nres);
    if (s == CUBLAS_STATUS_SUCCESS && nres < 1) s = CUBLAS_STATUS_NOT_SUPPORTED;
    if (s == CUBLAS_STATUS_SUCCESS) {
        if (dt == CUDA_R_64F) {
            double a = 1, b = 0;
            s = cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, C, Cd, D, Dd, &heur.algo, g_ws, g_ws_size, 0);
        } else {
            float a = 1, b = 0;
            s = cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, C, Cd, D, Dd, &heur.algo, g_ws, g_ws_size, 0);
        }
        cudaError_t ce = cudaDeviceSynchronize();
        if (ce != cudaSuccess) printf("   [cuda sync] %s\n", cudaGetErrorString(ce));
    }
    if (d_mant) {
        if (s == CUBLAS_STATUS_SUCCESS) cudaMemcpy(mantissa_out, d_mant, 4, cudaMemcpyDeviceToHost);
        cudaFree(d_mant);
    }
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
    cublasLtMatmulPreferenceDestroy(pref);
    if (emu) cublasLtEmulationDescDestroy(emu);
    cublasLtMatmulDescDestroy(op);
    return s;
}

int main() {
    cudaSetDevice(0);
    cudaMalloc(&g_ws, g_ws_size);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cublasHandle_t h; cublasCreate(&h);

    std::vector<float> hA(M * K), hB(K * N);
    fill(hA.data(), M * K, 1); fill(hB.data(), K * N, 2);
    std::vector<double> hAd(M * K), hBd(K * N);
    for (int i = 0; i < M * K; ++i) hAd[i] = hA[i];
    for (int i = 0; i < K * N; ++i) hBd[i] = hB[i];

    void *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, M * K * 8); cudaMalloc(&dB, K * N * 8);
    cudaMalloc(&dC, M * N * 8); cudaMalloc(&dD, M * N * 8);

    // ---------- f32 ----------
    cudaMemcpy(dA, hA.data(), M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K * N * 4, cudaMemcpyHostToDevice);
    std::vector<float> f32_native(M * N), f32_tf32(M * N), f32_bf16x9(M * N), f32_bf16x9_perf(M * N);
    cublasStatus_t s;
    s = run_lt(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, dA, dB, dC, dD, -1, nullptr);
    printf("f32 native            status=%d\n", (int)s);
    cudaMemcpy(f32_native.data(), dD, M * N * 4, cudaMemcpyDeviceToHost);
    s = run_lt(lt, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, dA, dB, dC, dD, -1, nullptr);
    printf("f32 tf32              status=%d\n", (int)s);
    cudaMemcpy(f32_tf32.data(), dD, M * N * 4, cudaMemcpyDeviceToHost);
    s = run_lt(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUDA_R_32F, dA, dB, dC, dD,
               CUBLAS_EMULATION_STRATEGY_EAGER, nullptr);
    printf("f32 bf16x9 eager      status=%d\n", (int)s);
    cudaMemcpy(f32_bf16x9.data(), dD, M * N * 4, cudaMemcpyDeviceToHost);
    s = run_lt(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUDA_R_32F, dA, dB, dC, dD,
               CUBLAS_EMULATION_STRATEGY_PERFORMANT, nullptr);
    printf("f32 bf16x9 performant status=%d\n", (int)s);
    cudaMemcpy(f32_bf16x9_perf.data(), dD, M * N * 4, cudaMemcpyDeviceToHost);

    auto cmp = [&](const char* name, const std::vector<float>& x, const std::vector<float>& y) {
        size_t same = 0; double num = 0, den = 0;
        for (int i = 0; i < M * N; ++i) {
            if (memcmp(&x[i], &y[i], 4) == 0) ++same;
            num += (double)(x[i] - y[i]) * (x[i] - y[i]);
            den += (double)x[i] * x[i];
        }
        printf("%-32s %8zu / %d identical, rel.diff=%9.2e\n", name, same, M * N, std::sqrt(num / den));
    };
    cmp("f32 native vs tf32", f32_native, f32_tf32);
    cmp("f32 native vs bf16x9(eager)", f32_native, f32_bf16x9);
    cmp("f32 native vs bf16x9(performant)", f32_native, f32_bf16x9_perf);
    cmp("bf16x9 eager vs performant", f32_bf16x9, f32_bf16x9_perf);

    // ---------- f64 ----------
    cudaMemcpy(dA, hAd.data(), M * K * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hBd.data(), K * N * 8, cudaMemcpyHostToDevice);
    std::vector<double> f64_native(M * N), f64_fixed(M * N), f64_fixed_perf(M * N);
    int32_t mant = -12345;
    s = run_lt(lt, CUBLAS_COMPUTE_64F, CUDA_R_64F, dA, dB, dC, dD, -1, nullptr);
    printf("f64 native            status=%d\n", (int)s);
    cudaMemcpy(f64_native.data(), dD, M * N * 8, cudaMemcpyDeviceToHost);
    s = run_lt(lt, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F, dA, dB, dC, dD,
               CUBLAS_EMULATION_STRATEGY_EAGER, &mant);
    printf("f64 fixed eager       status=%d mantissa_bits=%d (-1 = not used)\n", (int)s, mant);
    cudaMemcpy(f64_fixed.data(), dD, M * N * 8, cudaMemcpyDeviceToHost);
    int32_t mant2 = -12345;
    s = run_lt(lt, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F, dA, dB, dC, dD,
               CUBLAS_EMULATION_STRATEGY_PERFORMANT, &mant2);
    printf("f64 fixed performant  status=%d mantissa_bits=%d\n", (int)s, mant2);
    cudaMemcpy(f64_fixed_perf.data(), dD, M * N * 8, cudaMemcpyDeviceToHost);

    auto cmp_d = [&](const char* name, const std::vector<double>& x, const std::vector<double>& y) {
        size_t same = 0; double num = 0, den = 0;
        for (int i = 0; i < M * N; ++i) {
            if (memcmp(&x[i], &y[i], 8) == 0) ++same;
            num += (x[i] - y[i]) * (x[i] - y[i]);
            den += x[i] * x[i];
        }
        printf("%-32s %8zu / %d identical, rel.diff=%9.2e\n", name, same, M * N, std::sqrt(num / den));
    };
    cmp_d("f64 native vs fixed(eager)", f64_native, f64_fixed);
    cmp_d("f64 native vs fixed(performant)", f64_native, f64_fixed_perf);

    // ---------- classic math modes, bitwise ----------
    void *c1, *c2, *c3;
    cudaMalloc(&c1, M * N * 4); cudaMalloc(&c2, M * N * 4); cudaMalloc(&c3, M * N * 4);
    cudaMemcpy(dA, hA.data(), M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K * N * 4, cudaMemcpyHostToDevice);
    float al = 1, be = 0;
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &al, (float*)dA, M, (float*)dB, K, &be, (float*)c1, M);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &al, (float*)dA, M, (float*)dB, K, &be, (float*)c2, M);
    cublasSetMathMode(h, CUBLAS_FP32_EMULATED_BF16X9_MATH);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &al, (float*)dA, M, (float*)dB, K, &be, (float*)c3, M);
    cudaDeviceSynchronize();
    std::vector<float> cc1(M * N), cc2(M * N), cc3(M * N);
    cudaMemcpy(cc1.data(), c1, M * N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cc2.data(), c2, M * N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cc3.data(), c3, M * N * 4, cudaMemcpyDeviceToHost);
    cmp("Sgemm default vs TF32 math", cc1, cc2);
    cmp("Sgemm default vs BF16x9 math", cc1, cc3);

    return 0;
}
