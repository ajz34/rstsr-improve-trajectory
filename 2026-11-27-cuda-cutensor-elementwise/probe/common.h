// common.h -- shared helpers for the cuTENSOR elementwise probe and bench.
//
// Facts and timings only: every tensor is described explicitly by (extent, stride, offset) so the
// same helper serves contiguous, transposed, windowed, stepped, broadcast and strided-output cases.
#pragma once
#include <cutensor.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cd = std::complex<double>;

// ---------------------------------------------------------------------------------------- dtypes
struct DType {
    cudaDataType_t v;
    const char* name;
    int bytes;
    bool is_complex;
};

inline const DType DT_R16F{CUDA_R_16F, "R16F", 2, false};
inline const DType DT_R16BF{CUDA_R_16BF, "R16BF", 2, false};
inline const DType DT_R32F{CUDA_R_32F, "R32F", 4, false};
inline const DType DT_R64F{CUDA_R_64F, "R64F", 8, false};
inline const DType DT_C32F{CUDA_C_32F, "C32F", 8, true};
inline const DType DT_C64F{CUDA_C_64F, "C64F", 16, true};
inline const DType DT_R32I{CUDA_R_32I, "R32I", 4, false};
inline const DType DT_R64I{CUDA_R_64I, "R64I", 8, false};
inline const DType DT_R8U{CUDA_R_8U, "R8U", 1, false};
inline const DType DT_R8I{CUDA_R_8I, "R8I", 1, false};
inline const DType DT_R16I{CUDA_R_16I, "R16I", 2, false};
inline const DType DT_R32U{CUDA_R_32U, "R32U", 4, false};

inline std::string dt_name(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16F: return "R16F";
        case CUDA_R_16BF: return "R16BF";
        case CUDA_R_32F: return "R32F";
        case CUDA_R_64F: return "R64F";
        case CUDA_C_32F: return "C32F";
        case CUDA_C_64F: return "C64F";
        case CUDA_R_32I: return "R32I";
        case CUDA_R_64I: return "R64I";
        case CUDA_R_8U: return "R8U";
        case CUDA_R_8I: return "R8I";
        case CUDA_R_16I: return "R16I";
        case CUDA_R_32U: return "R32U";
        default: {
            char b[32];
            snprintf(b, sizeof b, "dt(%d)", (int)t);
            return b;
        }
    }
}

inline int dt_bytes(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_8U:
        case CUDA_R_8I: return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_16I: return 2;
        case CUDA_R_32F:
        case CUDA_R_32I:
        case CUDA_R_32U: return 4;
        case CUDA_R_64F:
        case CUDA_R_64I: return 8;
        case CUDA_C_32F: return 8;
        case CUDA_C_64F: return 16;
        default: return 0;
    }
}

inline bool dt_complex(cudaDataType_t t) { return t == CUDA_C_32F || t == CUDA_C_64F; }

// Compute descriptor that the documented table pairs with a storage type.
inline cutensorComputeDescriptor_t default_compute(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_32F:
        case CUDA_C_32F:
        case CUDA_R_32I:
        case CUDA_R_8U:
        case CUDA_R_8I:
        case CUDA_R_16I:
        case CUDA_R_32U: return CUTENSOR_COMPUTE_DESC_32F;
        case CUDA_R_64F:
        case CUDA_C_64F:
        case CUDA_R_64I: return CUTENSOR_COMPUTE_DESC_64F;
        default: return CUTENSOR_COMPUTE_DESC_32F;
    }
}

inline std::string comp_name(cutensorComputeDescriptor_t c) {
    if (c == CUTENSOR_COMPUTE_DESC_16F) return "16F";
    if (c == CUTENSOR_COMPUTE_DESC_16BF) return "16BF";
    if (c == CUTENSOR_COMPUTE_DESC_32F) return "32F";
    if (c == CUTENSOR_COMPUTE_DESC_64F) return "64F";
    if (c == CUTENSOR_COMPUTE_DESC_TF32) return "TF32";
    if (c == CUTENSOR_COMPUTE_DESC_3XTF32) return "3XTF32";
    return "other";
}

inline const char* st_name(cutensorStatus_t s) { return cutensorGetErrorString(s); }

// ------------------------------------------------------------------------------------- reporting
inline void emit(const char* group, const std::string& id, const std::string& status, const std::string& note = "") {
    printf("[%s] %-46s %-46s %s\n", group, id.c_str(), status.c_str(), note.c_str());
    fflush(stdout);
}

// ------------------------------------------------------------------------------------ device buf
struct DBuf {
    void* p = nullptr;
    size_t bytes = 0;
    DBuf() = default;
    explicit DBuf(size_t n) { alloc(n); }
    ~DBuf() {
        if (p) cudaFree(p);
    }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    DBuf(DBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
        o.p = nullptr;
        o.bytes = 0;
    }
    DBuf& operator=(DBuf&& o) noexcept {
        if (this != &o) {
            if (p) cudaFree(p);
            p = o.p;
            bytes = o.bytes;
            o.p = nullptr;
            o.bytes = 0;
        }
        return *this;
    }
    bool alloc(size_t n) {
        bytes = n;
        cudaError_t e = cudaMalloc(&p, std::max<size_t>(n, 256));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", n, cudaGetErrorString(e));
            p = nullptr;
            return false;
        }
        return true;
    }
    bool upload(const void* src, size_t n) {
        return cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool download(void* dst, size_t n) {
        return cudaMemcpy(dst, p, n, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
};

// ------------------------------------------------------------------------------------ tensor spec
struct TensorSpec {
    cudaDataType_t dt = CUDA_R_64F;
    std::vector<int64_t> ext;
    std::vector<int64_t> str;
    int64_t off = 0;
    int rank() const { return (int)ext.size(); }
    int64_t size() const {
        int64_t n = 1;
        for (auto e : ext) n *= e;
        return n;
    }
    // elements covered by the backing allocation [0, max_offset]; requires off + min_offset >= 0
    int64_t extent_elems() const {
        int64_t lo = 0, hi = 0;
        for (size_t i = 0; i < ext.size(); ++i) {
            int64_t a = (ext[i] - 1) * str[i];
            if (a < 0) lo += a; else hi += a;
        }
        return off + hi + 1;
    }
};

inline std::string spec_name(const TensorSpec& s) {
    std::string r = dt_name(s.dt) + "(";
    for (size_t i = 0; i < s.ext.size(); ++i) {
        char b[64];
        snprintf(b, sizeof b, "%s%lld:%lld", i ? "," : "", (long long)s.ext[i], (long long)s.str[i]);
        r += b;
    }
    r += ")";
    return r;
}

// ------------------------------------------------------------------------------- descriptor/plan
struct Plan {
    cutensorOperationDescriptor_t op = nullptr;
    cutensorPlanPreference_t pref = nullptr;
    cutensorPlan_t plan = nullptr;
    cutensorTensorDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    cudaDataType_t scalar = CUDA_R_32F;
    cutensorStatus_t s_op = CUTENSOR_STATUS_INTERNAL_ERROR;
    cutensorStatus_t s_pref = CUTENSOR_STATUS_INTERNAL_ERROR;
    cutensorStatus_t s_est = CUTENSOR_STATUS_INTERNAL_ERROR;
    cutensorStatus_t s_plan = CUTENSOR_STATUS_INTERNAL_ERROR;
    uint64_t ws = 0;
    int kind = 0;  // 1 binary, 2 trinary
    bool ok = false;
    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    Plan(Plan&& o) noexcept { *this = std::move(o); }
    Plan& operator=(Plan&& o) noexcept {
        if (this != &o) {
            reset();
            op = o.op; pref = o.pref; plan = o.plan;
            dA = o.dA; dB = o.dB; dC = o.dC; dD = o.dD;
            scalar = o.scalar;
            s_op = o.s_op; s_pref = o.s_pref; s_est = o.s_est; s_plan = o.s_plan;
            ws = o.ws; kind = o.kind; ok = o.ok;
            o.op = nullptr; o.pref = nullptr; o.plan = nullptr;
            o.dA = o.dB = o.dC = o.dD = nullptr;
            o.ok = false;
        }
        return *this;
    }
    ~Plan() { reset(); }
    void reset() {
        if (plan) cutensorDestroyPlan(plan);
        if (pref) cutensorDestroyPlanPreference(pref);
        if (op) cutensorDestroyOperationDescriptor(op);
        if (dA) cutensorDestroyTensorDescriptor(dA);
        if (dB) cutensorDestroyTensorDescriptor(dB);
        if (dC) cutensorDestroyTensorDescriptor(dC);
        if (dD) cutensorDestroyTensorDescriptor(dD);
        plan = nullptr; pref = nullptr; op = nullptr;
        dA = dB = dC = dD = nullptr;
        ok = false;
    }
};

inline cutensorStatus_t make_desc(cutensorHandle_t h, const TensorSpec& s, cutensorTensorDescriptor_t* out) {
    uint32_t align = (uint32_t)std::max(1, dt_bytes(s.dt));
    return cutensorCreateTensorDescriptor(h, out, (uint32_t)s.rank(), s.ext.empty() ? nullptr : s.ext.data(),
                                          s.str.empty() ? nullptr : s.str.data(), s.dt, align);
}

inline std::vector<int32_t> iota_modes(int n, int base = 0) {
    std::vector<int32_t> m(n);
    for (int i = 0; i < n; ++i) m[i] = base + i;
    return m;
}

inline void finish_plan(cutensorHandle_t h, Plan& p, cutensorComputeDescriptor_t comp, cutensorJitMode_t jit) {
    (void)comp;
    if (p.s_op != CUTENSOR_STATUS_SUCCESS || !p.op) {
        p.ok = false;
        return;
    }
    p.s_pref = cutensorCreatePlanPreference(h, &p.pref, CUTENSOR_ALGO_DEFAULT, jit);
    if (p.s_pref != CUTENSOR_STATUS_SUCCESS) {
        p.ok = false;
        return;
    }
    p.s_est = cutensorEstimateWorkspaceSize(h, p.op, p.pref, CUTENSOR_WORKSPACE_DEFAULT, &p.ws);
    if (p.s_est != CUTENSOR_STATUS_SUCCESS) {
        p.ok = false;
        return;
    }
    p.s_plan = cutensorCreatePlan(h, &p.plan, p.op, p.pref, p.ws);
    {
        cudaDataType_t st = CUDA_R_32F;
        if (cutensorOperationDescriptorGetAttribute(h, p.op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st,
                                                    sizeof(st)) == CUTENSOR_STATUS_SUCCESS) {
            p.scalar = st;
        }
    }
    p.ok = p.s_plan == CUTENSOR_STATUS_SUCCESS;
}

inline Plan make_binary_plan(cutensorHandle_t h, const TensorSpec& A, cutensorOperator_t opA,
                             const TensorSpec& CD, cutensorOperator_t opC, cutensorOperator_t opAC,
                             cutensorComputeDescriptor_t comp, cutensorJitMode_t jit = CUTENSOR_JIT_MODE_NONE,
                             const std::vector<int32_t>* modeA_arg = nullptr,
                             const std::vector<int32_t>* modeCD_arg = nullptr) {
    Plan p;
    p.kind = 1;
    std::vector<int32_t> modeA = modeA_arg ? *modeA_arg : iota_modes(A.rank());
    std::vector<int32_t> modeCD = modeCD_arg ? *modeCD_arg : iota_modes(CD.rank());
    p.s_op = make_desc(h, A, &p.dA);
    if (p.s_op == CUTENSOR_STATUS_SUCCESS) p.s_op = make_desc(h, CD, &p.dC);
    if (p.s_op == CUTENSOR_STATUS_SUCCESS) {
        p.s_op = cutensorCreateElementwiseBinary(h, &p.op, p.dA, modeA.data(), opA, p.dC, modeCD.data(), opC, p.dC,
                                                 modeCD.data(), opAC, comp);
    }
    p.dD = p.dC;  // shared descriptor; destroyed once
    finish_plan(h, p, comp, jit);
    return p;
}

inline Plan make_trinary_plan(cutensorHandle_t h, const TensorSpec& A, cutensorOperator_t opA, const TensorSpec& B,
                              cutensorOperator_t opB, const TensorSpec& CD, cutensorOperator_t opC,
                              cutensorOperator_t opAB, cutensorOperator_t opABC, cutensorComputeDescriptor_t comp,
                              cutensorJitMode_t jit = CUTENSOR_JIT_MODE_NONE,
                              const std::vector<int32_t>* modeA_arg = nullptr,
                              const std::vector<int32_t>* modeB_arg = nullptr,
                              const std::vector<int32_t>* modeCD_arg = nullptr) {
    Plan p;
    p.kind = 2;
    std::vector<int32_t> modeA = modeA_arg ? *modeA_arg : iota_modes(A.rank());
    std::vector<int32_t> modeB = modeB_arg ? *modeB_arg : iota_modes(B.rank());
    std::vector<int32_t> modeCD = modeCD_arg ? *modeCD_arg : iota_modes(CD.rank());
    p.s_op = make_desc(h, A, &p.dA);
    if (p.s_op == CUTENSOR_STATUS_SUCCESS) p.s_op = make_desc(h, B, &p.dB);
    if (p.s_op == CUTENSOR_STATUS_SUCCESS) p.s_op = make_desc(h, CD, &p.dC);
    if (p.s_op == CUTENSOR_STATUS_SUCCESS) {
        p.s_op = cutensorCreateElementwiseTrinary(h, &p.op, p.dA, modeA.data(), opA, p.dB, modeB.data(), opB, p.dC,
                                                  modeCD.data(), opC, p.dC, modeCD.data(), opAB, opABC, comp);
    }
    p.dD = p.dC;
    finish_plan(h, p, comp, jit);
    return p;
}

inline std::string plan_status(const Plan& p) {
    std::string s;
    s += "op=" + std::string(st_name(p.s_op));
    s += " pref=" + std::string(st_name(p.s_pref));
    s += " est=" + std::string(st_name(p.s_est));
    s += " plan=" + std::string(st_name(p.s_plan));
    return s;
}

// ---------------------------------------------------------------------------------------- scalars
inline void set_scalar(void* p, cudaDataType_t t, double re, double im = 0.0) {
    memset(p, 0, 16);
    switch (t) {
        case CUDA_R_32F: *(float*)p = (float)re; break;
        case CUDA_R_64F: *(double*)p = re; break;
        case CUDA_R_16F: *(__half*)p = __float2half_rn((float)re); break;
        case CUDA_R_16BF: *(__nv_bfloat16*)p = __float2bfloat16_rn((float)re); break;
        case CUDA_C_32F: *(cuComplex*)p = make_cuFloatComplex((float)re, (float)im); break;
        case CUDA_C_64F: *(cuDoubleComplex*)p = make_cuDoubleComplex(re, im); break;
        default: *(float*)p = (float)re; break;
    }
}

struct Scalars {
    alignas(16) uint8_t alpha[16];
    alignas(16) uint8_t beta[16];
    alignas(16) uint8_t gamma[16];
};

inline void fill_scalars(Scalars& s, cudaDataType_t t, double a, double b, double g) {
    set_scalar(s.alpha, t, a);
    set_scalar(s.beta, t, b);
    set_scalar(s.gamma, t, g);
}

inline cutensorStatus_t exec_binary(cutensorHandle_t h, const Plan& p, const Scalars& s, const void* A, void* C,
                                    void* D, cudaStream_t stream = nullptr) {
    return cutensorElementwiseBinaryExecute(h, p.plan, s.alpha, A, s.gamma, C, D, stream);
}

inline cutensorStatus_t exec_trinary(cutensorHandle_t h, const Plan& p, const Scalars& s, const void* A, const void* B,
                                     void* C, void* D, cudaStream_t stream = nullptr) {
    return cutensorElementwiseTrinaryExecute(h, p.plan, s.alpha, A, s.beta, B, s.gamma, C, D, stream);
}

// ------------------------------------------------------------------------------- host encode/decode
inline std::vector<uint8_t> encode(const std::vector<cd>& v, cudaDataType_t t) {
    std::vector<uint8_t> out(v.size() * (size_t)dt_bytes(t));
    for (size_t i = 0; i < v.size(); ++i) {
        cd z = v[i];
        switch (t) {
            case CUDA_R_16F: ((__half*)out.data())[i] = __float2half_rn((float)z.real()); break;
            case CUDA_R_16BF: ((__nv_bfloat16*)out.data())[i] = __float2bfloat16_rn((float)z.real()); break;
            case CUDA_R_32F: ((float*)out.data())[i] = (float)z.real(); break;
            case CUDA_R_64F: ((double*)out.data())[i] = z.real(); break;
            case CUDA_C_32F: ((cuComplex*)out.data())[i] = make_cuFloatComplex((float)z.real(), (float)z.imag()); break;
            case CUDA_C_64F: ((cuDoubleComplex*)out.data())[i] = make_cuDoubleComplex(z.real(), z.imag()); break;
            case CUDA_R_8U: ((uint8_t*)out.data())[i] = (uint8_t)(int64_t)std::llround(z.real()); break;
            case CUDA_R_8I: ((int8_t*)out.data())[i] = (int8_t)(int64_t)std::llround(z.real()); break;
            case CUDA_R_16I: ((int16_t*)out.data())[i] = (int16_t)(int64_t)std::llround(z.real()); break;
            case CUDA_R_32I: ((int32_t*)out.data())[i] = (int32_t)(int64_t)std::llround(z.real()); break;
            case CUDA_R_32U: ((uint32_t*)out.data())[i] = (uint32_t)(int64_t)std::llround(z.real()); break;
            case CUDA_R_64I: ((int64_t*)out.data())[i] = (int64_t)std::llround(z.real()); break;
            default: break;
        }
    }
    return out;
}

inline std::vector<cd> decode(const void* src, size_t n, cudaDataType_t t) {
    std::vector<cd> out(n);
    for (size_t i = 0; i < n; ++i) {
        switch (t) {
            case CUDA_R_16F: out[i] = cd(__half2float(((__half*)src)[i]), 0); break;
            case CUDA_R_16BF: out[i] = cd(__bfloat162float(((__nv_bfloat16*)src)[i]), 0); break;
            case CUDA_R_32F: out[i] = cd(((float*)src)[i], 0); break;
            case CUDA_R_64F: out[i] = cd(((double*)src)[i], 0); break;
            case CUDA_C_32F: {
                cuComplex z = ((cuComplex*)src)[i];
                out[i] = cd(z.x, z.y);
                break;
            }
            case CUDA_C_64F: {
                cuDoubleComplex z = ((cuDoubleComplex*)src)[i];
                out[i] = cd(z.x, z.y);
                break;
            }
            case CUDA_R_8U: out[i] = cd(((uint8_t*)src)[i], 0); break;
            case CUDA_R_8I: out[i] = cd(((int8_t*)src)[i], 0); break;
            case CUDA_R_16I: out[i] = cd(((int16_t*)src)[i], 0); break;
            case CUDA_R_32I: out[i] = cd(((int32_t*)src)[i], 0); break;
            case CUDA_R_32U: out[i] = cd(((uint32_t*)src)[i], 0); break;
            case CUDA_R_64I: out[i] = cd((double)((int64_t*)src)[i], 0); break;
            default: break;
        }
    }
    return out;
}

// logical row-major multi-index <-> linear offset in the strided arena
inline int64_t logical_to_offset(int64_t lin, const std::vector<int64_t>& ext, const std::vector<int64_t>& str,
                                 int64_t off) {
    int64_t o = off;
    for (int i = (int)ext.size() - 1; i >= 0; --i) {
        int64_t q = lin % ext[i];
        lin /= ext[i];
        o += q * str[i];
    }
    return o;
}

// ---------------------------------------------------------------------------------------- test TV
struct TV {
    TensorSpec spec;
    std::vector<cd> logical;
};

inline std::vector<uint8_t> stage(const TV& tv) {
    int64_t arena = std::max<int64_t>(tv.spec.extent_elems(), 1);
    std::vector<cd> buf((size_t)arena, cd(std::nan(""), std::nan("")));
    int64_t n = tv.spec.size();
    for (int64_t i = 0; i < n; ++i) {
        int64_t o = logical_to_offset(i, tv.spec.ext, tv.spec.str, tv.spec.off);
        if (o >= 0 && o < arena) buf[(size_t)o] = tv.logical[(size_t)i];
    }
    return encode(buf, tv.spec.dt);
}

inline std::vector<cd> fetch(const TV& tv, const void* dev) {
    int64_t arena = std::max<int64_t>(tv.spec.extent_elems(), 1);
    std::vector<uint8_t> bytes((size_t)arena * (size_t)dt_bytes(tv.spec.dt));
    cudaMemcpy(bytes.data(), dev, bytes.size(), cudaMemcpyDeviceToHost);
    std::vector<cd> buf = decode(bytes.data(), (size_t)arena, tv.spec.dt);
    std::vector<cd> out((size_t)tv.spec.size());
    for (int64_t i = 0; i < tv.spec.size(); ++i) {
        int64_t o = logical_to_offset(i, tv.spec.ext, tv.spec.str, tv.spec.off);
        out[(size_t)i] = (o >= 0 && o < arena) ? buf[(size_t)o] : cd(std::nan(""), std::nan(""));
    }
    return out;
}

// values as they actually sit in the staged arena (matters for overlapping/stride-0 stores)
inline std::vector<cd> placed_logical(const TV& tv, const std::vector<uint8_t>& staged) {
    int64_t arena = std::max<int64_t>(tv.spec.extent_elems(), 1);
    std::vector<cd> buf = decode(staged.data(), (size_t)arena, tv.spec.dt);
    std::vector<cd> out((size_t)tv.spec.size());
    for (int64_t i = 0; i < tv.spec.size(); ++i)
        out[(size_t)i] = buf[(size_t)logical_to_offset(i, tv.spec.ext, tv.spec.str, tv.spec.off)];
    return out;
}

// output logical index -> operand logical linear index under numpy broadcasting
inline int64_t bcast_index(int64_t out_lin, const std::vector<int64_t>& out_ext, const std::vector<int64_t>& in_ext) {
    int64_t idx[16];
    for (int i = (int)out_ext.size() - 1; i >= 0; --i) {
        idx[i] = out_lin % out_ext[i];
        out_lin /= out_ext[i];
    }
    int64_t in_lin = 0;
    for (int i = 0; i < (int)out_ext.size(); ++i) {
        int64_t ie = (i < (int)in_ext.size()) ? in_ext[i] : 1;
        in_lin = in_lin * ie + (ie == 1 ? 0 : idx[i]);
    }
    return in_lin;
}

// ------------------------------------------------------------------------------------- references
inline cd ref_unary(cutensorOperator_t op, cd x) {
    switch (op) {
        case CUTENSOR_OP_IDENTITY: return x;
        case CUTENSOR_OP_NEG: return -x;
        case CUTENSOR_OP_CONJ: return std::conj(x);
        case CUTENSOR_OP_ABS: return cd(std::abs(x), 0);
        case CUTENSOR_OP_EXP: return std::exp(x);
        case CUTENSOR_OP_LOG: return std::log(x);
        case CUTENSOR_OP_SQRT: return std::sqrt(x);
        case CUTENSOR_OP_RCP: return cd(1.0) / x;
        case CUTENSOR_OP_CEIL: return cd(std::ceil(x.real()), 0);
        case CUTENSOR_OP_FLOOR: return cd(std::floor(x.real()), 0);
        case CUTENSOR_OP_SIN: return std::sin(x);
        case CUTENSOR_OP_COS: return std::cos(x);
        case CUTENSOR_OP_TANH: return std::tanh(x);
        case CUTENSOR_OP_RELU: return cd(std::max(x.real(), 0.0), 0);
        case CUTENSOR_OP_SIGMOID: return cd(1.0 / (1.0 + std::exp(-x.real())), 0);
        default: return cd(std::nan(""), std::nan(""));
    }
}

inline cd ref_binary(cutensorOperator_t op, cd a, cd b) {
    switch (op) {
        case CUTENSOR_OP_ADD: return a + b;
        case CUTENSOR_OP_MUL: return a * b;
        case CUTENSOR_OP_MAX:
            if (a.imag() != 0 || b.imag() != 0) return cd(std::nan(""), 0);
            return cd(std::max(a.real(), b.real()), 0);
        case CUTENSOR_OP_MIN:
            if (a.imag() != 0 || b.imag() != 0) return cd(std::nan(""), 0);
            return cd(std::min(a.real(), b.real()), 0);
        default: return cd(std::nan(""), std::nan(""));
    }
}

inline bool close(cd a, cd b, double rtol, double atol) {
    if (std::isnan(a.real()) && std::isnan(b.real())) return std::isnan(a.imag()) == std::isnan(b.imag());
    double d = std::abs(a - b);
    double m = std::max(std::abs(a), std::abs(b));
    return d <= atol + rtol * m;
}

inline double tol_for(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2e-2;
        case CUDA_R_32F:
        case CUDA_C_32F: return 2e-5;
        case CUDA_R_64F:
        case CUDA_C_64F: return 1e-12;
        default: return 0.0;
    }
}

// deterministic input values, positive and negative, away from exp overflow
inline std::vector<cd> make_vals(size_t n, uint64_t seed, bool complex, bool positive_only = false) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.25, 1.25);
    std::vector<cd> v(n);
    for (size_t i = 0; i < n; ++i) {
        double re = u(rng);
        if (!positive_only && (i % 3 == 1)) re = -re;
        double im = complex ? u(rng) : 0.0;
        v[i] = cd(re, im);
    }
    return v;
}

// ---------------------------------------------------------------------------------------- timing
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct EventTimer {
    cudaEvent_t a = nullptr, b = nullptr;
    EventTimer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~EventTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void start(cudaStream_t s = nullptr) { cudaEventRecord(a, s); }
    double stop_ms(cudaStream_t s = nullptr) {
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};
