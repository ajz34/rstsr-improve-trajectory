// Classic cuBLAS handle-level emulation controls (no environment variables).
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static const int M = 512, N = 512, K = 512;

static void ref_gemm(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    C.assign((size_t)M*N, 0.0);
    for (int j = 0; j < N; ++j)
        for (int p = 0; p < K; ++p) {
            double b = B[(size_t)p + (size_t)j*K];
            const double* a = &A[(size_t)p*M];
            double* c = &C[(size_t)j*M];
            for (int i = 0; i < M; ++i) c[i] += a[i]*b;
        }
}

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    void* ws = nullptr; cudaMalloc(&ws, 64ull<<20); cublasSetWorkspace(h, ws, 64ull<<20);

    std::vector<double> hA(M*K), hB(K*N);
    unsigned s = 1;
    for (auto& v : hA) { s = s*1664525u+1013904223u; v = (double)((int)(s>>8)%2001-1000)/1000.0; }
    for (auto& v : hB) { s = s*1664525u+1013904223u; v = (double)((int)(s>>8)%2001-1000)/1000.0; }
    std::vector<double> ref; ref_gemm(hA, hB, ref);

    void *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M*K*8); cudaMalloc(&dB, (size_t)K*N*8); cudaMalloc(&dC, (size_t)M*N*8);
    cudaMemcpy(dA, hA.data(), (size_t)M*K*8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), (size_t)K*N*8, cudaMemcpyHostToDevice);

    std::vector<double> native_out;
    auto run = [&](const char* name, cublasMath_t mode, int strategy, int mantissa_bits) {
        cublasSetEmulationStrategy(h, (cublasEmulationStrategy_t)strategy);
        cublasSetFixedPointEmulationMantissaControl(h,
            mantissa_bits > 0 ? CUDA_EMULATION_MANTISSA_CONTROL_FIXED : CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
        if (mantissa_bits > 0) cublasSetFixedPointEmulationMaxMantissaBitCount(h, mantissa_bits);
        cublasSetMathMode(h, mode);
        double a = 1, b = 0;
        cublasStatus_t st = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &a, (double*)dA, M,
                                        (double*)dB, K, &b, (double*)dC, M);
        cudaDeviceSynchronize();
        std::vector<double> out((size_t)M*N);
        cudaMemcpy(out.data(), dC, out.size()*8, cudaMemcpyDeviceToHost);
        double num = 0, den = 0; size_t same = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            num += (out[i]-ref[i])*(out[i]-ref[i]);
            den += ref[i]*ref[i];
            if (!native_out.empty() && memcmp(&out[i], &native_out[i], 8) == 0) ++same;
        }
        if (native_out.empty()) native_out = out;
        printf("%-46s status=%2d rel.err=%.2e  bit-identical-to-native=%zu/%zu\n",
               name, (int)st, std::sqrt(num/den), same, out.size());
    };

    run("f64 default math", CUBLAS_DEFAULT_MATH, 0, 0);
    run("fixed-point math, dynamic, default strategy", CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, 0, 0);
    run("fixed-point math, dynamic, eager", CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
        CUBLAS_EMULATION_STRATEGY_EAGER, 0);
    run("fixed-point math, fixed 30 bits, eager", CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
        CUBLAS_EMULATION_STRATEGY_EAGER, 30);
    run("fixed-point math, fixed 40 bits, eager", CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
        CUBLAS_EMULATION_STRATEGY_EAGER, 40);
    run("fixed-point math, fixed 55 bits, eager", CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
        CUBLAS_EMULATION_STRATEGY_EAGER, 55);

    // same for FP32 BF16x9
    std::vector<float> fA(M*K), fB(K*N);
    for (int i = 0; i < M*K; ++i) fA[i] = (float)hA[i];
    for (int i = 0; i < K*N; ++i) fB[i] = (float)hB[i];
    void *eA, *eB, *eC;
    cudaMalloc(&eA, (size_t)M*K*4); cudaMalloc(&eB, (size_t)K*N*4); cudaMalloc(&eC, (size_t)M*N*4);
    cudaMemcpy(eA, fA.data(), (size_t)M*K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(eB, fB.data(), (size_t)K*N*4, cudaMemcpyHostToDevice);
    cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetMathMode(h, CUBLAS_FP32_EMULATED_BF16X9_MATH);
    float af = 1, bf = 0;
    cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &af, (float*)eA, M, (float*)eB, K, &bf, (float*)eC, M);
    cudaDeviceSynchronize();
    printf("Sgemm BF16x9 math, eager: status=%d\n", (int)st);
    return 0;
}
