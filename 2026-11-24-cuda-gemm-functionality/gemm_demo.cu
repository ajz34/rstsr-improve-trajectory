// GEMM functionality matrix for cuBLAS / cuBLASLt on the local GPU.
//
// For every (API, input dtype, output dtype, compute mode) combination this
// program runs one small GEMM and reports
//   * whether cuBLAS accepted the combination,
//   * the relative Frobenius error against an exact double-precision
//     reference (or the exact-integer reference for int8).
//
// No timing is done on purpose: the question is "what can be expressed",
// not "what is fast".
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120 gemm_demo.cu -o gemm_demo \
//            -lcublas -lcublasLt

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static const char* sname(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "OK";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        default: return "STATUS_OTHER";
    }
}

static size_t esize(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16F: case CUDA_R_16BF: return 2;
        case CUDA_R_32F: case CUDA_R_32I: return 4;
        case CUDA_R_64F: return 8;
        case CUDA_C_32F: return 8;
        case CUDA_C_64F: return 16;
        case CUDA_R_8I: case CUDA_R_8F_E4M3: case CUDA_R_8F_E5M2: return 1;
        case CUDA_R_4F_E2M1: return 1;
        default: return 0;
    }
}

static bool is_quantized_input(cudaDataType_t t) {
    return t == CUDA_R_16F || t == CUDA_R_16BF || t == CUDA_R_8F_E4M3 || t == CUDA_R_8F_E5M2;
}

struct Rng {
    uint64_t s = 88172645463325252ull;
    double next() {  // uniform in [-1, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return double(s >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    }
    int next_i8() {  // uniform in [-8, 7]
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return int((s >> 33) % 16) - 8;
    }
};

// ---------------------------------------------------------------------------
// Host packing / unpacking
// ---------------------------------------------------------------------------

static void pack(std::vector<unsigned char>& out, cudaDataType_t t,
                 const std::vector<double>& re, const std::vector<double>& im,
                 size_t n) {
    out.assign(n * esize(t), 0);
    for (size_t i = 0; i < n; ++i) {
        switch (t) {
            case CUDA_R_16F: { __half h = __float2half((float)re[i]); memcpy(&out[i * 2], &h, 2); break; }
            case CUDA_R_16BF: { __nv_bfloat16 h = __float2bfloat16((float)re[i]); memcpy(&out[i * 2], &h, 2); break; }
            case CUDA_R_32F: { float f = (float)re[i]; memcpy(&out[i * 4], &f, 4); break; }
            case CUDA_R_64F: { double d = re[i]; memcpy(&out[i * 8], &d, 8); break; }
            case CUDA_R_8F_E4M3: { __nv_fp8_e4m3 x((float)re[i]); memcpy(&out[i], &x, 1); break; }
            case CUDA_R_8F_E5M2: { __nv_fp8_e5m2 x((float)re[i]); memcpy(&out[i], &x, 1); break; }
            case CUDA_C_32F: {
                float f0 = (float)re[i], f1 = (float)im[i];
                memcpy(&out[i * 8], &f0, 4); memcpy(&out[i * 8 + 4], &f1, 4); break;
            }
            case CUDA_C_64F: {
                double d0 = re[i], d1 = im[i];
                memcpy(&out[i * 16], &d0, 8); memcpy(&out[i * 16 + 8], &d1, 8); break;
            }
            case CUDA_R_8I: { int8_t x = (int8_t)re[i]; memcpy(&out[i], &x, 1); break; }
            case CUDA_R_32I: { int32_t x = (int32_t)re[i]; memcpy(&out[i * 4], &x, 4); break; }
            default: break;
        }
    }
}

static double unpack_real(const unsigned char* p, cudaDataType_t t, size_t i) {
    switch (t) {
        case CUDA_R_16F: { __half h; memcpy(&h, p + i * 2, 2); return __half2float(h); }
        case CUDA_R_16BF: { __nv_bfloat16 h; memcpy(&h, p + i * 2, 2); return __bfloat162float(h); }
        case CUDA_R_32F: { float f; memcpy(&f, p + i * 4, 4); return f; }
        case CUDA_R_64F: { double d; memcpy(&d, p + i * 8, 8); return d; }
        case CUDA_R_8F_E4M3: { __nv_fp8_e4m3 x; memcpy(&x, p + i, 1); return (float)x; }
        case CUDA_R_8F_E5M2: { __nv_fp8_e5m2 x; memcpy(&x, p + i, 1); return (float)x; }
        case CUDA_R_8I: { int8_t x; memcpy(&x, p + i, 1); return x; }
        case CUDA_R_32I: { int32_t x; memcpy(&x, p + i * 4, 4); return x; }
        default: return 0.0;
    }
}

static void unpack_complex(const unsigned char* p, cudaDataType_t t, size_t i,
                           double& re, double& im) {
    if (t == CUDA_C_32F) {
        float f0, f1; memcpy(&f0, p + i * 8, 4); memcpy(&f1, p + i * 8 + 4, 4);
        re = f0; im = f1;
    } else {
        double d0, d1; memcpy(&d0, p + i * 16, 8); memcpy(&d1, p + i * 16 + 8, 8);
        re = d0; im = d1;
    }
}

// ---------------------------------------------------------------------------
// Reference GEMMs (column-major)
// ---------------------------------------------------------------------------

static void ref_gemm(int m, int n, int k,
                     const std::vector<double>& A, const std::vector<double>& B,
                     std::vector<double>& C) {
    C.assign((size_t)m * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p) {
            const double b = B[(size_t)p + (size_t)j * k];
            const double* a = &A[(size_t)p * m];
            double* c = &C[(size_t)j * m];
            for (int i = 0; i < m; ++i) c[i] += a[i] * b;
        }
}

static void ref_gemm_c(int m, int n, int k,
                       const std::vector<std::complex<double>>& A,
                       const std::vector<std::complex<double>>& B,
                       std::vector<std::complex<double>>& C) {
    C.assign((size_t)m * n, {0.0, 0.0});
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p) {
            const std::complex<double> b = B[(size_t)p + (size_t)j * k];
            const std::complex<double>* a = &A[(size_t)p * m];
            std::complex<double>* c = &C[(size_t)j * m];
            for (int i = 0; i < m; ++i) c[i] += a[i] * b;
        }
}

static void ref_gemm_i8(int m, int n, int k,
                        const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                        std::vector<int64_t>& C) {
    C.assign((size_t)m * n, 0);
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p) {
            const int64_t b = B[(size_t)p + (size_t)j * k];
            const int8_t* a = &A[(size_t)p * m];
            int64_t* c = &C[(size_t)j * m];
            for (int i = 0; i < m; ++i) c[i] += (int64_t)a[i] * b;
        }
}

// ---------------------------------------------------------------------------
// Case description
// ---------------------------------------------------------------------------

enum class Api { Lt, GemmEx, Typed, MathMode };

struct Case {
    const char* group;
    const char* label;
    Api api;
    cudaDataType_t at, bt, ct, dt;
    cublasComputeType_t compute;
    cudaDataType_t scale_type;
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    float scale_a = 1.0f, scale_b = 1.0f;
    bool eager_emulation = false;
    bool complex = false;
    bool int8 = false;
    int mantissa_bits = 0;  // classic fixed-point emulation with fixed mantissa
};

static Case mk(const char* g, const char* l, Api api, cudaDataType_t at,
               cudaDataType_t bt, cudaDataType_t ct, cudaDataType_t dt,
               cublasComputeType_t comp, cudaDataType_t st,
               cublasMath_t mm = CUBLAS_DEFAULT_MATH,
               float sa = 1.0f, float sb = 1.0f, bool eager = false, int mantissa_bits = 0) {
    Case c{};
    c.group = g; c.label = l; c.api = api;
    c.at = at; c.bt = bt; c.ct = ct; c.dt = dt;
    c.compute = comp; c.scale_type = st; c.math_mode = mm;
    c.scale_a = sa; c.scale_b = sb; c.eager_emulation = eager;
    c.mantissa_bits = mantissa_bits;
    return c;
}

// ---------------------------------------------------------------------------
// Device buffers
// ---------------------------------------------------------------------------

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes ? bytes : 1); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

static void upload(void* dst, const std::vector<unsigned char>& h) {
    cudaMemcpy(dst, h.data(), h.size(), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// Launch paths
// ---------------------------------------------------------------------------

static cublasStatus_t launch_lt(cublasLtHandle_t lt, const Case& c,
                                const void* dA, const void* dB, void* dC, void* dD,
                                int m, int n, int k) {
    cublasLtMatmulDesc_t op = nullptr;
    cublasStatus_t s = cublasLtMatmulDescCreate(&op, c.compute, c.scale_type);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    DevBuf* sa = nullptr; DevBuf* sb = nullptr;
    if (c.scale_a != 1.0f || c.scale_b != 1.0f) {
        sa = new DevBuf(sizeof(float)); sb = new DevBuf(sizeof(float));
        cudaMemcpy(sa->p, &c.scale_a, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(sb->p, &c.scale_b, sizeof(float), cudaMemcpyHostToDevice);
        void* pa = sa->p; void* pb = sb->p;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb));
    }

    cublasLtEmulationDesc_t emu = nullptr;
    if (c.eager_emulation) {
        if (cublasLtEmulationDescCreate(&emu) == CUBLAS_STATUS_SUCCESS) {
            int32_t strategy = CUBLAS_EMULATION_STRATEGY_EAGER;
            cublasLtEmulationDescSetAttribute(emu, CUBLASLT_EMULATION_DESC_STRATEGY,
                                              &strategy, sizeof(strategy));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                           &emu, sizeof(emu));
        }
    }

    cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
    cublasLtMatrixLayoutCreate(&Ad, c.at, m, k, m);
    cublasLtMatrixLayoutCreate(&Bd, c.bt, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, c.ct, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, c.dt, m, n, m);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 64ull << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t heur;
    int nres = 0;
    s = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &nres);
    if (s == CUBLAS_STATUS_SUCCESS && nres < 1) s = CUBLAS_STATUS_NOT_SUPPORTED;

    DevBuf* workspace = nullptr;
    if (s == CUBLAS_STATUS_SUCCESS) {
        workspace = new DevBuf(ws);
        if (c.scale_type == CUDA_R_16F) {
            __half a = __float2half(1.0f), b = __float2half(0.0f);
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        } else if (c.scale_type == CUDA_C_32F) {
            cuComplex a = make_cuComplex(1.0f, 0.0f), b = make_cuComplex(0.0f, 0.0f);
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        } else if (c.scale_type == CUDA_C_64F) {
            cuDoubleComplex a = make_cuDoubleComplex(1.0, 0.0), b = make_cuDoubleComplex(0.0, 0.0);
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        } else if (c.compute == CUBLAS_COMPUTE_32I) {
            int32_t a = 1, b = 0;
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        } else if (c.scale_type == CUDA_R_64F) {
            double a = 1.0, b = 0.0;
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        } else {
            float a = 1.0f, b = 0.0f;
            s = cublasLtMatmul(lt, op, &a, dA, Ad, dB, Bd, &b, dC, Cd, dD, Dd,
                               &heur.algo, workspace->p, ws, 0);
        }
    }

    delete workspace;
    if (emu) cublasLtEmulationDescDestroy(emu);
    delete sa; delete sb;
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(op);
    return s;
}

static cublasStatus_t launch_gemmex(cublasHandle_t h, const Case& c,
                                    const void* dA, const void* dB, void* dD,
                                    int m, int n, int k) {
    __half ah = __float2half(1.0f), bh = __float2half(0.0f);
    float af = 1.0f, bf = 0.0f;
    double ad = 1.0, bd = 0.0;
    int32_t ai = 1, bi = 0;
    const void* alpha = &af; const void* beta = &bf;
    if (c.scale_type == CUDA_R_16F) { alpha = &ah; beta = &bh; }
    else if (c.scale_type == CUDA_R_64F) { alpha = &ad; beta = &bd; }
    else if (c.scale_type == CUDA_R_32I) { alpha = &ai; beta = &bi; }

    return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                        alpha, dA, c.at, m, dB, c.bt, k, beta, dD, c.dt, m,
                        c.compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static cublasStatus_t launch_typed(cublasHandle_t h, const Case& c,
                                   const void* dA, const void* dB, void* dD,
                                   int m, int n, int k) {
    if (c.dt == CUDA_R_32F) {
        float a = 1.0f, b = 0.0f;
        return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                           (const float*)dA, m, (const float*)dB, k, &b, (float*)dD, m);
    }
    if (c.dt == CUDA_R_64F) {
        double a = 1.0, b = 0.0;
        return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                           (const double*)dA, m, (const double*)dB, k, &b, (double*)dD, m);
    }
    if (c.dt == CUDA_R_16F) {
        __half a = __float2half(1.0f), b = __float2half(0.0f);
        return cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                           (const __half*)dA, m, (const __half*)dB, k, &b, (__half*)dD, m);
    }
    if (c.dt == CUDA_C_32F) {
        cuComplex a = make_cuComplex(1.0f, 0.0f), b = make_cuComplex(0.0f, 0.0f);
        return cublasCgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                           (const cuComplex*)dA, m, (const cuComplex*)dB, k, &b, (cuComplex*)dD, m);
    }
    if (c.dt == CUDA_C_64F) {
        cuDoubleComplex a = make_cuDoubleComplex(1.0, 0.0), b = make_cuDoubleComplex(0.0, 0.0);
        return cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                           (const cuDoubleComplex*)dA, m, (const cuDoubleComplex*)dB, k,
                           &b, (cuDoubleComplex*)dD, m);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

static cublasStatus_t launch_mathmode(cublasHandle_t h, const Case& c,
                                      const void* dA, const void* dB, void* dD,
                                      int m, int n, int k) {
    if (c.mantissa_bits > 0) {
        cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER);
        cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
        cublasSetFixedPointEmulationMaxMantissaBitCount(h, c.mantissa_bits);
    }
    cublasSetMathMode(h, c.math_mode);
    cublasStatus_t s;
    if (c.dt == CUDA_R_32F) {
        float a = 1.0f, b = 0.0f;
        s = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                        (const float*)dA, m, (const float*)dB, k, &b, (float*)dD, m);
    } else {
        double a = 1.0, b = 0.0;
        s = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a,
                        (const double*)dA, m, (const double*)dB, k, &b, (double*)dD, m);
    }
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (c.mantissa_bits > 0) {
        cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_DEFAULT);
        cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
    }
    return s;
}

// ---------------------------------------------------------------------------
// One case: run + accuracy
// ---------------------------------------------------------------------------

struct Result {
    bool ran = false;
    double err_exact = 0.0;
    double err_quant = 0.0;
    double max_abs_q = 0.0;
    long mismatches = -1;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
};

static Result run_case(cublasHandle_t h, cublasLtHandle_t lt, const Case& c,
                       const std::vector<double>& Are, const std::vector<double>& Aim,
                       const std::vector<double>& Bre, const std::vector<double>& Bim,
                       const std::vector<double>& Cref_re, const std::vector<double>& Cref_im,
                       int m, int n, int k) {
    Result r;
    std::vector<unsigned char> ha, hb, hc;
    pack(ha, c.at, Are, Aim, (size_t)m * k);
    pack(hb, c.bt, Bre, Bim, (size_t)k * n);
    std::vector<double> zero((size_t)m * n, 0.0);
    pack(hc, c.ct, zero, zero, (size_t)m * n);

    DevBuf dA(ha.size()), dB(hb.size()), dC(hc.size()), dD((size_t)m * n * esize(c.dt));
    upload(dA.p, ha); upload(dB.p, hb); upload(dC.p, hc);
    cudaMemset(dD.p, 0, (size_t)m * n * esize(c.dt));

    switch (c.api) {
        case Api::Lt: r.status = launch_lt(lt, c, dA.p, dB.p, dC.p, dD.p, m, n, k); break;
        case Api::GemmEx: r.status = launch_gemmex(h, c, dA.p, dB.p, dD.p, m, n, k); break;
        case Api::Typed: r.status = launch_typed(h, c, dA.p, dB.p, dD.p, m, n, k); break;
        case Api::MathMode: r.status = launch_mathmode(h, c, dA.p, dB.p, dD.p, m, n, k); break;
    }
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
        fprintf(stderr, "  [cuda] %s: %s\n", c.label, cudaGetErrorString(ce));
        r.status = CUBLAS_STATUS_EXECUTION_FAILED;
        return r;
    }
    if (r.status != CUBLAS_STATUS_SUCCESS) return r;

    std::vector<unsigned char> hd((size_t)m * n * esize(c.dt));
    cudaMemcpy(hd.data(), dD.p, hd.size(), cudaMemcpyDeviceToHost);

    // Reference built from the values actually stored (after dtype quantization).
    std::vector<double> Cref_quant;
    if (!c.complex && (is_quantized_input(c.at) || is_quantized_input(c.bt))) {
        std::vector<double> qa((size_t)m * k), qb((size_t)k * n);
        for (size_t i = 0; i < qa.size(); ++i) qa[i] = unpack_real(ha.data(), c.at, i);
        for (size_t i = 0; i < qb.size(); ++i) qb[i] = unpack_real(hb.data(), c.bt, i);
        ref_gemm(m, n, k, qa, qb, Cref_quant);
    } else {
        Cref_quant = Cref_re;
    }
    const double s = (double)c.scale_a * (double)c.scale_b;

    double num_e = 0, num_q = 0, den_e = 0, den_q = 0;
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        double re, im;
        if (c.complex) {
            unpack_complex(hd.data(), c.dt, i, re, im);
        } else {
            re = unpack_real(hd.data(), c.dt, i);
            im = 0.0;
        }
        const double ere = Cref_re[i] * s;
        const double eim = c.complex ? Cref_im[i] * s : 0.0;
        const double qre = Cref_quant[i] * s;
        const double qim = c.complex ? Cref_im[i] * s : 0.0;
        const double de = re - ere, di = im - eim;
        const double qe = re - qre, qi = im - qim;
        num_e += de * de + di * di;
        num_q += qe * qe + qi * qi;
        if (std::fabs(qe) > r.max_abs_q) r.max_abs_q = std::fabs(qe);
        if (std::fabs(qi) > r.max_abs_q) r.max_abs_q = std::fabs(qi);
        den_e += ere * ere + eim * eim;
        den_q += qre * qre + qim * qim;
    }
    r.ran = true;
    r.err_exact = den_e > 0 ? std::sqrt(num_e / den_e) : 0.0;
    r.err_quant = den_q > 0 ? std::sqrt(num_q / den_q) : 0.0;
    return r;
}

static Result run_int8(cublasHandle_t h, cublasLtHandle_t lt, const Case& c,
                       const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                       const std::vector<int64_t>& Cref, int m, int n, int k) {
    Result r;
    std::vector<double> re(A.begin(), A.end()), im(A.size(), 0.0);
    std::vector<double> bre(B.begin(), B.end()), bim(B.size(), 0.0);
    std::vector<unsigned char> ha, hb, hc;
    pack(ha, c.at, re, im, (size_t)m * k);
    pack(hb, c.bt, bre, bim, (size_t)k * n);
    std::vector<double> zero((size_t)m * n, 0.0);
    pack(hc, c.ct, zero, zero, (size_t)m * n);

    DevBuf dA(ha.size()), dB(hb.size()), dC(hc.size()), dD((size_t)m * n * esize(c.dt));
    upload(dA.p, ha); upload(dB.p, hb); upload(dC.p, hc);
    cudaMemset(dD.p, 0, (size_t)m * n * esize(c.dt));

    r.status = (c.api == Api::Lt) ? launch_lt(lt, c, dA.p, dB.p, dC.p, dD.p, m, n, k)
                                  : launch_gemmex(h, c, dA.p, dB.p, dD.p, m, n, k);
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
        fprintf(stderr, "  [cuda] %s: %s\n", c.label, cudaGetErrorString(ce));
        r.status = CUBLAS_STATUS_EXECUTION_FAILED;
        return r;
    }
    if (r.status != CUBLAS_STATUS_SUCCESS) return r;

    std::vector<int32_t> hd((size_t)m * n);
    cudaMemcpy(hd.data(), dD.p, hd.size() * 4, cudaMemcpyDeviceToHost);
    r.ran = true;
    r.mismatches = 0;
    for (size_t i = 0; i < (size_t)m * n; ++i)
        if ((int64_t)hd[i] != Cref[i]) ++r.mismatches;
    return r;
}

// Strided-batched GEMM: one launch, four independent f32 products.
static bool demo_batched(cublasHandle_t h) {
    const int b = 4, mm = 128, nn = 128, kk = 128;
    const size_t sa = (size_t)mm * kk, sb = (size_t)kk * nn, sc = (size_t)mm * nn;
    std::vector<float> A(sa * b, 0.5f), B(sb * b, 0.25f), C(sc * b, 0.0f);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sa * b * 4); cudaMalloc(&dB, sb * b * 4); cudaMalloc(&dC, sc * b * 4);
    cudaMemcpy(dA, A.data(), sa * b * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sb * b * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), sc * b * 4, cudaMemcpyHostToDevice);
    float alpha = 1, beta = 0;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, mm, nn, kk, &alpha,
        dA, CUDA_R_32F, mm, (long long)sa,
        dB, CUDA_R_32F, kk, (long long)sb,
        &beta, dC, CUDA_R_32F, mm, (long long)sc,
        b, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    bool ok = s == CUBLAS_STATUS_SUCCESS;
    if (ok) {
        cudaMemcpy(C.data(), dC, sc * b * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < b; ++i)
            if (std::fabs(C[(size_t)i * sc] - 0.5f * 0.25f * kk) > 1e-3f) ok = false;
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cublasHandle_t h;
    cublasCreate(&h);
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    void* wsptr = nullptr;
    cudaMalloc(&wsptr, 64ull << 20);
    cublasSetWorkspace(h, wsptr, 64ull << 20);

    int cublas_ver = 0;
    cublasGetVersion(h, &cublas_ver);
    size_t lt_ver = cublasLtGetVersion();

    printf("GPU            : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("cuBLAS         : %d, cuBLASLt: %zu\n", cublas_ver, lt_ver);
    printf("CUDA runtime   : %d.%d, driver API %d.%d\n",
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("\n");

    const int R = 512;
    const int C = 256;

    // ---- data -------------------------------------------------------------
    Rng rng;
    std::vector<double> Are(R * R), Aim(R * R, 0.0), Bre(R * R), Bim(R * R, 0.0);
    for (auto& v : Are) v = rng.next();
    for (auto& v : Bre) v = rng.next();

    std::vector<double> Ac_re(C * C), Ac_im(C * C), Bc_re(C * C), Bc_im(C * C);
    std::vector<std::complex<double>> Ac(C * C), Bc(C * C);
    for (size_t i = 0; i < (size_t)C * C; ++i) {
        Ac_re[i] = rng.next(); Ac_im[i] = rng.next();
        Bc_re[i] = rng.next(); Bc_im[i] = rng.next();
        Ac[i] = {Ac_re[i], Ac_im[i]};
        Bc[i] = {Bc_re[i], Bc_im[i]};
    }

    std::vector<int8_t> Ai(R * R), Bi(R * R);
    for (auto& v : Ai) v = (int8_t)rng.next_i8();
    for (auto& v : Bi) v = (int8_t)rng.next_i8();

    std::vector<double> Cref_real;
    ref_gemm(R, R, R, Are, Bre, Cref_real);

    std::vector<std::complex<double>> Cref_c;
    ref_gemm_c(C, C, C, Ac, Bc, Cref_c);
    std::vector<double> Cref_c_re(C * C), Cref_c_im(C * C);
    for (size_t i = 0; i < Cref_c.size(); ++i) {
        Cref_c_re[i] = Cref_c[i].real();
        Cref_c_im[i] = Cref_c[i].imag();
    }

    std::vector<int64_t> Cref_i;
    ref_gemm_i8(R, R, R, Ai, Bi, Cref_i);

    // The reference is stored as two planes [re][im] for complex; make a
    // combined array so run_case can index both.
    // ---- cases ------------------------------------------------------------
    std::vector<Case> cases;

    cases.push_back(mk("classic", "cublasSgemm", Api::Typed, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("classic", "cublasDgemm", Api::Typed, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    cases.push_back(mk("classic", "cublasHgemm", Api::Typed, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    {
        Case c = mk("classic", "cublasCgemm", Api::Typed, CUDA_C_32F, CUDA_C_32F,
                    CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F, CUDA_C_32F);
        c.complex = true; cases.push_back(c);
        Case z = mk("classic", "cublasZgemm", Api::Typed, CUDA_C_64F, CUDA_C_64F,
                    CUDA_C_64F, CUDA_C_64F, CUBLAS_COMPUTE_64F, CUDA_C_64F);
        z.complex = true; cases.push_back(z);
    }

    cases.push_back(mk("classic", "Sgemm + TF32 math", Api::MathMode, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                       CUBLAS_TF32_TENSOR_OP_MATH));
    cases.push_back(mk("classic", "Sgemm + BF16x9 math", Api::MathMode, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                       CUBLAS_FP32_EMULATED_BF16X9_MATH));
    cases.push_back(mk("classic", "Dgemm + fixed-point math", Api::MathMode, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                       CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    cases.push_back(mk("classic", "Dgemm fixed-point, 30 bits", Api::MathMode, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                       CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, 1.f, 1.f, false, 30));
    cases.push_back(mk("classic", "Dgemm fixed-point, 40 bits", Api::MathMode, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                       CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, 1.f, 1.f, false, 40));
    cases.push_back(mk("classic", "Dgemm fixed-point, 55 bits", Api::MathMode, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                       CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, 1.f, 1.f, false, 55));

    cases.push_back(mk("GemmEx", "f16 -> f32, compute 32F", Api::GemmEx, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("GemmEx", "bf16 -> f32, compute 32F", Api::GemmEx, CUDA_R_16BF, CUDA_R_16BF,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("GemmEx", "f16 -> f16, compute 16F", Api::GemmEx, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    cases.push_back(mk("GemmEx", "f32, compute TF32", Api::GemmEx, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cases.push_back(mk("GemmEx", "f64, compute 64F", Api::GemmEx, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F));

    cases.push_back(mk("cuBLASLt", "f32, compute 32F", Api::Lt, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f32, compute TF32", Api::Lt, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f32, compute 16BFX9 (eager)", Api::Lt, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUDA_R_32F,
                       CUBLAS_DEFAULT_MATH, 1.f, 1.f, true));
    cases.push_back(mk("cuBLASLt", "f32, compute FAST_16F", Api::Lt, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f32, compute FAST_16BF", Api::Lt, CUDA_R_32F, CUDA_R_32F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f64, compute 64F", Api::Lt, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    cases.push_back(mk("cuBLASLt", "f64, compute fixed-point (eager)", Api::Lt, CUDA_R_64F, CUDA_R_64F,
                       CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F,
                       CUBLAS_DEFAULT_MATH, 1.f, 1.f, true));
    cases.push_back(mk("cuBLASLt", "f16 -> f32, compute 32F", Api::Lt, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f16 -> f16, compute 16F", Api::Lt, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    cases.push_back(mk("cuBLASLt", "f16 -> f16, compute 32F", Api::Lt, CUDA_R_16F, CUDA_R_16F,
                       CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "bf16 -> f32, compute 32F", Api::Lt, CUDA_R_16BF, CUDA_R_16BF,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "bf16 -> bf16, compute 32F", Api::Lt, CUDA_R_16BF, CUDA_R_16BF,
                       CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "bf16 -> bf16, compute 16F", Api::Lt, CUDA_R_16BF, CUDA_R_16BF,
                       CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    cases.push_back(mk("cuBLASLt", "e4m3 -> f32, scale 0.5/2", Api::Lt, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                       CUBLAS_DEFAULT_MATH, 0.5f, 2.0f));
    cases.push_back(mk("cuBLASLt", "e4m3 -> f16, no scale", Api::Lt, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
                       CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "e5m2 -> f32, no scale", Api::Lt, CUDA_R_8F_E5M2, CUDA_R_8F_E5M2,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "e2m1 (fp4) -> f32", Api::Lt, CUDA_R_4F_E2M1, CUDA_R_4F_E2M1,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cases.push_back(mk("cuBLASLt", "f16 + bf16 mixed -> f32", Api::Lt, CUDA_R_16F, CUDA_R_16BF,
                       CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    {
        Case c = mk("cuBLASLt", "c32 -> c32, compute 32F", Api::Lt, CUDA_C_32F, CUDA_C_32F,
                    CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F, CUDA_C_32F);
        c.complex = true; cases.push_back(c);
        Case z = mk("cuBLASLt", "c64 -> c64, compute 64F", Api::Lt, CUDA_C_64F, CUDA_C_64F,
                    CUDA_C_64F, CUDA_C_64F, CUBLAS_COMPUTE_64F, CUDA_C_64F);
        z.complex = true; cases.push_back(z);
    }
    {
        Case c = mk("GemmEx", "i8 -> i32, compute 32I", Api::GemmEx, CUDA_R_8I, CUDA_R_8I,
                    CUDA_R_32I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        c.int8 = true; cases.push_back(c);
        Case l = mk("cuBLASLt", "i8 -> i32, compute 32I", Api::Lt, CUDA_R_8I, CUDA_R_8I,
                    CUDA_R_32I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        l.int8 = true; cases.push_back(l);
    }

    // ---- run --------------------------------------------------------------
    int ok = 0, bad = 0;
    printf("%-9s %-34s %-13s %-13s %-10s %s\n", "API", "case", "result", "err/exact", "err/stored", "max|d|");
    printf("%s\n", std::string(104, '-').c_str());
    for (const Case& c : cases) {
        Result r;
        if (c.int8) {
            r = run_int8(h, lt, c, Ai, Bi, Cref_i, R, R, R);
        } else if (c.complex) {
            r = run_case(h, lt, c, Ac_re, Ac_im, Bc_re, Bc_im, Cref_c_re, Cref_c_im, C, C, C);
        } else {
            std::vector<double> no_im;
            r = run_case(h, lt, c, Are, Aim, Bre, Bim, Cref_real, no_im, R, R, R);
        }

        if (!r.ran) {
            printf("%-9s %-34s %-13s\n", c.group, c.label, sname(r.status));
            ++bad;
            continue;
        }
        if (c.int8) {
            printf("%-9s %-34s %-13s mismatches=%ld\n", c.group, c.label, "OK", r.mismatches);
        } else {
            printf("%-9s %-34s %-13s %-13.2e %-10.2e %.2e\n", c.group, c.label, "OK",
                   r.err_exact, r.err_quant, r.max_abs_q);
        }
        ++ok;
    }

    printf("%-9s %-34s %-13s %s\n", "classic", "strided-batched f32 (b=4)",
           demo_batched(h) ? "OK" : "FAILED", "4 independent 128^3 products");
    printf("%s\n", std::string(104, '-').c_str());
    printf("accepted: %d   rejected: %d   (out of %zu)\n", ok, bad, cases.size());

    cublasDestroy(h);
    cublasLtDestroy(lt);
    cudaFree(wsptr);
    return 0;
}
