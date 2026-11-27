// Shared framework for the cuTENSOR probe: buffers, descriptors, status-tracked
// contraction pipeline, CPU reference, reporting.
#pragma once

#include <cutensor.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using W = std::complex<double>;
using CF = std::complex<float>;
using clk = std::chrono::steady_clock;

#define CUDA_CHECK(x)                                                                                                   \
    do {                                                                                                                \
        cudaError_t e_ = (x);                                                                                           \
        if (e_ != cudaSuccess) {                                                                                        \
            fflush(stdout);                                                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d (%s)\n", cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                    cudaGetErrorString(e_));                                                                            \
            exit(2);                                                                                                    \
        }                                                                                                               \
    } while (0)

static const char* sname(cutensorStatus_t s) { return cutensorGetErrorString(s); }
static bool isok(cutensorStatus_t s) { return s == CUTENSOR_STATUS_SUCCESS; }

static size_t eszOf(cudaDataType_t dt) {
    switch (dt) {
        case CUDA_R_64F:
        case CUDA_C_32F: return 8;
        case CUDA_C_64F: return 16;
        case CUDA_R_32F: return 4;
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        default: fprintf(stderr, "unsupported dtype %d\n", (int)dt); exit(3);
    }
}
static bool isCplx(cudaDataType_t dt) { return dt == CUDA_C_64F || dt == CUDA_C_32F; }
static const char* dtName(cudaDataType_t dt) {
    switch (dt) {
        case CUDA_R_64F: return "f64";
        case CUDA_C_64F: return "c64";
        case CUDA_R_32F: return "f32";
        case CUDA_C_32F: return "c32";
        case CUDA_R_16F: return "f16";
        case CUDA_R_16BF: return "bf16";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// Tensor spec: modes, extents, strides (elements), dtype, alignment, base offset.
struct TSpec {
    std::vector<int32_t> modes;
    std::vector<int64_t> extent, stride;
    cudaDataType_t dt = CUDA_R_64F;
    uint32_t align = 256;
    int64_t baseOff = 0;  // offset (elements) of the tensor's base inside its buffer
    TSpec() {}
    TSpec(std::vector<int32_t> m, std::vector<int64_t> e, std::vector<int64_t> s, cudaDataType_t d = CUDA_R_64F,
          uint32_t a = 256)
        : modes(m), extent(e), stride(s), dt(d), align(a) {}
    size_t rank() const { return modes.size(); }
    // buffer elements needed: elements live at baseOff + sum(idx_i*stride_i); the caller must choose
    // baseOff so that the minimum (negative strides) stays >= 0
    size_t span() const {
        int64_t lo = baseOff, hi = baseOff;
        for (size_t i = 0; i < extent.size(); ++i) {
            if (extent[i] <= 0) continue;
            int64_t d = (extent[i] - 1) * stride[i];
            hi += std::max<int64_t>(d, 0);
            lo += std::min<int64_t>(d, 0);
        }
        if (lo < 0) {
            fprintf(stderr, "TSpec: baseOff %lld too small for negative strides (min %lld)\n", (long long)baseOff,
                    (long long)lo);
            exit(3);
        }
        return (size_t)(hi + 1);
    }
};

// packed column-major spec
static TSpec colMajor(std::vector<int32_t> modes, std::vector<int64_t> ext, cudaDataType_t dt = CUDA_R_64F,
                      uint32_t a = 256) {
    std::vector<int64_t> st(ext.size());
    int64_t acc = 1;
    for (size_t i = 0; i < ext.size(); ++i) {
        st[i] = acc;
        acc *= ext[i];
    }
    return TSpec(modes, ext, st, dt, a);
}

template <class T>
static T ld(const uint8_t* p) {
    T v;
    memcpy(&v, p, sizeof(T));
    return v;
}
template <class T>
static void st(uint8_t* p, T v) {
    memcpy(p, &v, sizeof(T));
}

static W readElem(const uint8_t* base, cudaDataType_t dt, int64_t off) {
    const uint8_t* p = base + off * (int64_t)eszOf(dt);
    switch (dt) {
        case CUDA_R_64F: return W(ld<double>(p), 0.0);
        case CUDA_C_64F: {
            cuDoubleComplex z = ld<cuDoubleComplex>(p);
            return W(z.x, z.y);
        }
        case CUDA_R_32F: return W(ld<float>(p), 0.0);
        case CUDA_C_32F: {
            cuFloatComplex z = ld<cuFloatComplex>(p);
            return W(z.x, z.y);
        }
        case CUDA_R_16F: return W((double)__half2float(ld<__half>(p)), 0.0);
        case CUDA_R_16BF: return W((double)__bfloat162float(ld<__nv_bfloat16>(p)), 0.0);
        default: return W(0, 0);
    }
}
static void writeElem(uint8_t* base, cudaDataType_t dt, int64_t off, W v) {
    uint8_t* p = base + off * (int64_t)eszOf(dt);
    switch (dt) {
        case CUDA_R_64F: st<double>(p, v.real()); break;
        case CUDA_C_64F: {
            cuDoubleComplex z = make_cuDoubleComplex(v.real(), v.imag());
            st<cuDoubleComplex>(p, z);
            break;
        }
        case CUDA_R_32F: st<float>(p, (float)v.real()); break;
        case CUDA_C_32F: {
            cuFloatComplex z = make_cuFloatComplex((float)v.real(), (float)v.imag());
            st<cuFloatComplex>(p, z);
            break;
        }
        case CUDA_R_16F: st<__half>(p, __double2half(v.real())); break;
        case CUDA_R_16BF: st<__nv_bfloat16>(p, __double2bfloat16(v.real())); break;
        default: break;
    }
}

static W randW(cudaDataType_t dt, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double re = u(rng);
    if (std::fabs(re) < 0.05) re += 0.3;  // avoid tiny magnitudes
    if (isCplx(dt)) {
        double im = u(rng);
        if (std::fabs(im) < 0.05) im -= 0.3;
        return W(re, im);
    }
    return W(re, 0);
}

// ---------------------------------------------------------------------------
// Device buffer with host mirror (h0 = initial content, h = after execution).
struct Buf {
    cudaDataType_t dt = CUDA_R_64F;
    size_t n = 0;
    int64_t baseOff = 0;
    void* d = nullptr;
    std::vector<uint8_t> h0, h;
    size_t esz() const { return eszOf(dt); }
    void* dptr() const { return (char*)d + baseOff * (int64_t)esz(); }
};

struct Tsr {
    TSpec s;
    Buf* buf = nullptr;
    bool own = false;
    Tsr() {}
    // allocate + fill (random, or constant sentinel)
    Tsr(const TSpec& spec, std::mt19937_64& rng, bool random = true, W sentinel = W(-7.5e300)) : s(spec) {
        buf = new Buf();
        own = true;
        buf->dt = spec.dt;
        buf->n = spec.span();
        buf->baseOff = spec.baseOff;
        buf->h0.resize(buf->n * eszOf(spec.dt));
        for (size_t i = 0; i < buf->n; ++i) writeElem(buf->h0.data(), spec.dt, (int64_t)i, random ? randW(spec.dt, rng) : sentinel);
        CUDA_CHECK(cudaMalloc(&buf->d, buf->n * eszOf(spec.dt)));
        CUDA_CHECK(cudaMemcpy(buf->d, buf->h0.data(), buf->n * eszOf(spec.dt), cudaMemcpyHostToDevice));
    }
    // alias another tensor's storage (e.g. C == D in-place)
    Tsr(const TSpec& spec, const Tsr& other) : s(spec), buf(other.buf), own(false) {}
    ~Tsr() {
        if (own) {
            cudaFree(buf->d);
            delete buf;
        }
    }
    void download() {
        buf->h = buf->h0;
        CUDA_CHECK(cudaMemcpy(buf->h.data(), buf->d, buf->n * buf->esz(), cudaMemcpyDeviceToHost));
    }
};

// ---------------------------------------------------------------------------
// Status-tracked pipeline
struct Stages {
    cutensorStatus_t desc[4] = {CUTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS,
                                CUTENSOR_STATUS_SUCCESS};  // A, B, C, D
    cutensorStatus_t op = CUTENSOR_STATUS_SUCCESS, pref = CUTENSOR_STATUS_SUCCESS,
                     ws = CUTENSOR_STATUS_SUCCESS, plan = CUTENSOR_STATUS_SUCCESS, exec = CUTENSOR_STATUS_SUCCESS;
    uint64_t wsEstimate = 0, wsRequired = 0;
    std::string hookFailStage;
    std::string firstFail() const {
        if (!hookFailStage.empty()) return hookFailStage;
        const char* nm[4] = {"descA", "descB", "descC", "descD"};
        for (int i = 0; i < 4; ++i)
            if (!isok(desc[i])) return nm[i];
        if (!isok(op)) return "CreateContraction";
        if (!isok(pref)) return "CreatePlanPreference";
        if (!isok(ws)) return "EstimateWorkspaceSize";
        if (!isok(plan)) return "CreatePlan";
        if (!isok(exec)) return "Contract";
        return "";
    }
    cutensorStatus_t firstFailStatus() const {
        for (int i = 0; i < 4; ++i)
            if (!isok(desc[i])) return desc[i];
        if (!isok(op)) return op;
        if (!isok(pref)) return pref;
        if (!isok(ws)) return ws;
        if (!isok(plan)) return plan;
        return exec;
    }
    bool ok() const { return firstFail().empty(); }
};

static const int64_t* arrOrNull(const std::vector<int64_t>& v) { return v.empty() ? nullptr : v.data(); }

struct PlanKeep {
    cutensorOperationDescriptor_t op = nullptr;
    cutensorPlan_t plan = nullptr;
    ~PlanKeep() {
        if (plan) cutensorDestroyPlan(plan);
        if (op) cutensorDestroyOperationDescriptor(op);
    }
};

// Optional hook, invoked with the operation descriptor right after a successful
// cutensorCreateContraction (before planning). It may overwrite st.op to stop the pipeline.
using OpHook = std::function<void(cutensorOperationDescriptor_t, Stages&)>;

// Runs the full pipeline; records every status; never aborts on library errors.
static void runContraction(cutensorHandle_t h, Tsr& A, cutensorOperator_t opA, Tsr& B, cutensorOperator_t opB, Tsr& C,
                           cutensorOperator_t opC, Tsr& D, cutensorComputeDescriptor_t comp, const void* alpha,
                           const void* beta, void* ws, uint64_t wsSize, cudaStream_t stream, Stages& st,
                           PlanKeep* keep = nullptr, int cacheMode = -1, bool execute = true, OpHook opHook = nullptr) {
    cutensorTensorDescriptor_t dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
    cutensorOperationDescriptor_t op = nullptr;
    cutensorPlanPreference_t pref = nullptr;
    cutensorPlan_t plan = nullptr;
    st.desc[0] = cutensorCreateTensorDescriptor(h, &dA, (uint32_t)A.s.rank(), arrOrNull(A.s.extent), arrOrNull(A.s.stride), A.s.dt, A.s.align);
    st.desc[1] = cutensorCreateTensorDescriptor(h, &dB, (uint32_t)B.s.rank(), arrOrNull(B.s.extent), arrOrNull(B.s.stride), B.s.dt, B.s.align);
    st.desc[2] = cutensorCreateTensorDescriptor(h, &dC, (uint32_t)C.s.rank(), arrOrNull(C.s.extent), arrOrNull(C.s.stride), C.s.dt, C.s.align);
    st.desc[3] = cutensorCreateTensorDescriptor(h, &dD, (uint32_t)D.s.rank(), arrOrNull(D.s.extent), arrOrNull(D.s.stride), D.s.dt, D.s.align);
    if (isok(st.desc[0]) && isok(st.desc[1]) && isok(st.desc[2]) && isok(st.desc[3])) {
        st.op = cutensorCreateContraction(h, &op, dA, A.s.modes.empty() ? nullptr : A.s.modes.data(), opA, dB,
                                          B.s.modes.empty() ? nullptr : B.s.modes.data(), opB, dC,
                                          C.s.modes.empty() ? nullptr : C.s.modes.data(), opC, dD,
                                          D.s.modes.empty() ? nullptr : D.s.modes.data(), comp);
    }
    if (isok(st.op) && opHook) opHook(op, st);
    if (isok(st.op)) {
        st.pref = cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
        if (isok(st.pref) && cacheMode >= 0) {
            int32_t cm = cacheMode;
            cutensorStatus_t s = cutensorPlanPreferenceSetAttribute(h, pref, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cm, sizeof(cm));
            if (!isok(s)) st.pref = s;
        }
    }
    if (isok(st.pref)) st.ws = cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &st.wsEstimate);
    if (isok(st.ws)) st.plan = cutensorCreatePlan(h, &plan, op, pref, wsSize);
    if (isok(st.plan)) {
        cutensorPlanGetAttribute(h, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &st.wsRequired, sizeof(st.wsRequired));
        if (execute) {
            st.exec = cutensorContract(h, plan, alpha, A.buf->dptr(), B.buf->dptr(), beta, C.buf->dptr(), D.buf->dptr(), ws,
                                       wsSize, stream);
            if (isok(st.exec)) {
                cudaError_t ce = cudaStreamSynchronize(stream);
                if (ce != cudaSuccess) {
                    fflush(stdout);
                    fprintf(stderr, "   cuda error after cutensorContract: %s\n", cudaGetErrorString(ce));
                    st.exec = CUTENSOR_STATUS_CUDA_ERROR;
                }
            }
        }
    }
    if (keep && isok(st.op)) {
        keep->op = op;
        op = nullptr;
        keep->plan = plan;
        plan = nullptr;
    }
    if (plan) cutensorDestroyPlan(plan);
    if (op) cutensorDestroyOperationDescriptor(op);
    if (pref) cutensorDestroyPlanPreference(pref);
    if (dA) cutensorDestroyTensorDescriptor(dA);
    if (dB) cutensorDestroyTensorDescriptor(dB);
    if (dC) cutensorDestroyTensorDescriptor(dC);
    if (dD) cutensorDestroyTensorDescriptor(dD);
}

// ---------------------------------------------------------------------------
// CPU reference: D = alpha * opA(A) * opB(B) + beta * opC(C), accumulated in ACC.
template <class ACC>
static ACC conv(W v) {
    if constexpr (std::is_same<ACC, double>::value || std::is_same<ACC, float>::value) return (ACC)v.real();
    else return ACC((typename ACC::value_type)v.real(), (typename ACC::value_type)v.imag());
}
template <class ACC>
static W toW(ACC v) {
    if constexpr (std::is_same<ACC, double>::value || std::is_same<ACC, float>::value) return W((double)v, 0.0);
    else return W((double)v.real(), (double)v.imag());
}
template <class ACC>
static ACC cconj(ACC v) {
    if constexpr (std::is_same<ACC, double>::value || std::is_same<ACC, float>::value) return v;
    else return std::conj(v);
}

static std::map<int32_t, int64_t> modeExtents(const std::vector<TSpec*>& specs) {
    std::map<int32_t, int64_t> ext;
    for (auto* s : specs)
        for (size_t i = 0; i < s->modes.size(); ++i) {
            if (ext.count(s->modes[i]) && ext[s->modes[i]] != s->extent[i])
                fprintf(stderr, "   WARNING: inconsistent extent for mode %d\n", s->modes[i]);
            ext[s->modes[i]] = s->extent[i];
        }
    return ext;
}

template <class ACC>
static void refContract(const uint8_t* Ah, const TSpec& sA, bool conjA, const uint8_t* Bh, const TSpec& sB, bool conjB,
                        ACC alpha, ACC beta, const uint8_t* Ch, const TSpec& sC, bool conjC, uint8_t* Dh,
                        const TSpec& sD, const std::map<int32_t, int64_t>& ext) {
    std::vector<int32_t> outModes = sD.modes, redModes;
    auto has = [](const std::vector<int32_t>& v, int32_t m) { return std::find(v.begin(), v.end(), m) != v.end(); };
    for (int32_t m : sA.modes)
        if (!has(outModes, m) && !has(redModes, m)) redModes.push_back(m);
    for (int32_t m : sB.modes)
        if (!has(outModes, m) && !has(redModes, m)) redModes.push_back(m);
    std::vector<int32_t> all = outModes;
    all.insert(all.end(), redModes.begin(), redModes.end());
    const int nOut = (int)outModes.size(), nAll = (int)all.size();
    auto posOf = [&](const TSpec& s) {
        std::vector<int> p(s.modes.size());
        for (size_t i = 0; i < s.modes.size(); ++i) p[i] = (int)(std::find(all.begin(), all.end(), s.modes[i]) - all.begin());
        return p;
    };
    std::vector<int> pA = posOf(sA), pB = posOf(sB), pC = posOf(sC), pD = posOf(sD);
    // per-tensor strides in 'all' order (+= so that a duplicated mode sums its strides, i.e. a diagonal)
    std::vector<int64_t> sAa(nAll, 0), sBa(nAll, 0), sCa(nAll, 0), sDa(nAll, 0), extAll(nAll);
    for (size_t i = 0; i < sA.modes.size(); ++i) sAa[pA[i]] += sA.stride[i];
    for (size_t i = 0; i < sB.modes.size(); ++i) sBa[pB[i]] += sB.stride[i];
    for (size_t i = 0; i < sC.modes.size(); ++i) sCa[pC[i]] += sC.stride[i];
    for (size_t i = 0; i < sD.modes.size(); ++i) sDa[pD[i]] += sD.stride[i];
    for (int i = 0; i < nAll; ++i) extAll[i] = ext.at(all[i]);
    const int64_t bA = sA.baseOff, bB = sB.baseOff, bC = sC.baseOff, bD = sD.baseOff;
    int64_t nOutElems = 1, nRedElems = 1;
    for (int i = 0; i < nOut; ++i) nOutElems *= extAll[i];
    for (int i = nOut; i < nAll; ++i) nRedElems *= extAll[i];
    const bool useC = beta != ACC(0);
#pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < nOutElems; ++o) {
        std::vector<int64_t> idx(nAll);
        int64_t q = o;
        for (int i = 0; i < nOut; ++i) {
            idx[i] = q % extAll[i];
            q /= extAll[i];
        }
        int64_t oA = bA, oB = bB, oC = bC, oD = bD;
        for (int i = 0; i < nOut; ++i) {
            oA += idx[i] * sAa[i];
            oB += idx[i] * sBa[i];
            oC += idx[i] * sCa[i];
            oD += idx[i] * sDa[i];
        }
        ACC acc(0);
        for (int64_t r = 0; r < nRedElems; ++r) {
            int64_t dA = 0, dB = 0;
            q = r;
            for (int i = nOut; i < nAll; ++i) {
                int64_t j = q % extAll[i];
                q /= extAll[i];
                dA += j * sAa[i];
                dB += j * sBa[i];
            }
            ACC a = conv<ACC>(readElem(Ah, sA.dt, oA + dA));
            ACC b = conv<ACC>(readElem(Bh, sB.dt, oB + dB));
            if (conjA) a = cconj(a);
            if (conjB) b = cconj(b);
            acc += a * b;
        }
        ACC dv = alpha * acc;
        if (useC) {
            ACC c = conv<ACC>(readElem(Ch, sC.dt, oC));
            if (conjC) c = cconj(c);
            dv += beta * c;
        }
        writeElem(Dh, sD.dt, oD, toW(dv));
    }
}

// Offsets of all elements addressed by a spec (relative to its base).
static std::vector<int64_t> touchedOffsets(const TSpec& s, const std::map<int32_t, int64_t>& ext) {
    std::vector<int64_t> out;
    const int n = (int)s.modes.size();
    for (int i = 0; i < n; ++i)
        if (ext.at(s.modes[i]) <= 0) return out;
    std::vector<int64_t> idx(n, 0);
    for (;;) {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) o += idx[i] * s.stride[i];
        out.push_back(o);
        int i = n - 1;
        for (; i >= 0; --i) {
            if (++idx[i] < ext.at(s.modes[i])) break;
            idx[i] = 0;
        }
        if (i < 0) break;
    }
    return out;
}

// ---------------------------------------------------------------------------
struct Scalars {
    const void* alpha = nullptr;
    const void* beta = nullptr;
    W alphaV{1, 0}, betaV{0, 0};
    std::string desc;
};
template <class T>
static T scalarOf(W v) {
    if constexpr (std::is_same<T, cuDoubleComplex>::value) return make_cuDoubleComplex(v.real(), v.imag());
    else if constexpr (std::is_same<T, cuFloatComplex>::value) return make_cuFloatComplex((float)v.real(), (float)v.imag());
    else return (T)v.real();
}
template <class T>
static Scalars mkScalars(W a, W b, const char* what) {
    T* pa = new T(scalarOf<T>(a));
    T* pb = new T(scalarOf<T>(b));
    Scalars s;
    s.alpha = pa;
    s.beta = pb;
    s.alphaV = a;
    s.betaV = b;
    char buf[256];
    snprintf(buf, sizeof buf, "alpha/beta passed as %s", what);
    s.desc = buf;
    return s;
}
static Scalars scalarsF64(double a, double b) { return mkScalars<double>(W(a, 0), W(b, 0), "double"); }
static Scalars scalarsF32(float a, float b) { return mkScalars<float>(W(a, 0), W(b, 0), "float"); }
static Scalars scalarsC64(W a, W b) { return mkScalars<cuDoubleComplex>(a, b, "cuDoubleComplex"); }
static Scalars scalarsC32(W a, W b) { return mkScalars<cuFloatComplex>(a, b, "cuFloatComplex"); }

struct CheckResult {
    bool anyErr = false;
    std::string errStage, errStr;
    int errCode = 0;
    bool haveResult = false;
    double maxAbs = 0, maxRel = 0, maxRef = 0;
    bool untouchedChecked = false, untouchedOk = true;
    size_t nTouched = 0, nUntouched = 0;
    Stages st;
};

static double tolOf(cudaDataType_t dt) {
    switch (dt) {
        case CUDA_R_64F:
        case CUDA_C_64F: return 1e-12;
        case CUDA_R_32F:
        case CUDA_C_32F: return 1e-5;
        case CUDA_R_16F: return 1e-2;
        case CUDA_R_16BF: return 8e-2;
        default: return 1e-12;
    }
}

// Runs the GPU pipeline, then compares against the CPU reference (ACC accumulation).
template <class ACC>
static CheckResult doCheck(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, Tsr& A,
                           cutensorOperator_t opA, Tsr& B, cutensorOperator_t opB, Tsr& C, cutensorOperator_t opC, Tsr& D,
                            cutensorComputeDescriptor_t comp, const Scalars& sc, bool checkUntouched, int cacheMode = -1,
                           OpHook opHook = nullptr) {
    CheckResult r;
    runContraction(h, A, opA, B, opB, C, opC, D, comp, sc.alpha, sc.beta, ws, wsSize, stream, r.st, nullptr, cacheMode, true,
                   opHook);
    if (!r.st.ok()) {
        r.anyErr = true;
        r.errStage = r.st.firstFail();
        cutensorStatus_t s = r.st.firstFailStatus();
        r.errCode = (int)s;
        char buf[256];
        snprintf(buf, sizeof buf, "status=%d (%s)", (int)s, sname(s));
        r.errStr = buf;
        return r;
    }
    D.download();
    std::vector<TSpec*> specs = {&A.s, &B.s, &C.s, &D.s};
    auto ext = modeExtents(specs);
    std::vector<uint8_t> href = D.buf->h0;
    refContract<ACC>(A.buf->h0.data(), A.s, opA == CUTENSOR_OP_CONJ, B.buf->h0.data(), B.s, opB == CUTENSOR_OP_CONJ,
                     conv<ACC>(sc.alphaV), conv<ACC>(sc.betaV), C.buf->h0.data(), C.s, opC == CUTENSOR_OP_CONJ,
                     href.data(), D.s, ext);
    std::vector<int64_t> offs = touchedOffsets(D.s, ext);
    std::vector<uint8_t> touched(D.buf->n, 0);
    const size_t es = D.buf->esz();
    for (int64_t o : offs) {
        size_t i = (size_t)(o + D.buf->baseOff);
        touched[i] = 1;
        W got = readElem(D.buf->h.data(), D.s.dt, (int64_t)i);
        W ref = readElem(href.data(), D.s.dt, (int64_t)i);
        double e = std::abs(got - ref);
        r.maxAbs = std::max(r.maxAbs, e);
        if (std::abs(ref) > 0) r.maxRel = std::max(r.maxRel, e / std::abs(ref));
        r.maxRef = std::max(r.maxRef, std::abs(ref));
    }
    r.nTouched = offs.size();
    r.haveResult = true;
    if (getenv("PROBE_DEBUG")) {
        for (size_t j = 0; j < std::min<size_t>(3, offs.size()); ++j) {
            size_t i = (size_t)(offs[j] + D.buf->baseOff);
            W got = readElem(D.buf->h.data(), D.s.dt, (int64_t)i), ref = readElem(href.data(), D.s.dt, (int64_t)i);
            W c = readElem(C.buf->h0.data(), C.s.dt, (int64_t)i);
            printf("   DEBUG %s elem %zu: gpu=%g ref=%g C=%g betaV=%g betaPtr=%p\n", "D", j, got.real(), ref.real(), c.real(),
                   sc.betaV.real(), (const void*)sc.beta);
        }
        double* bp = (double*)sc.beta;
        printf("   DEBUG *beta(double*) = %g, *alpha = %g\n", bp[0], ((double*)sc.alpha)[0]);
    }
    if (checkUntouched) {
        r.untouchedChecked = true;
        for (size_t i = 0; i < D.buf->n; ++i) {
            if (touched[i]) continue;
            ++r.nUntouched;
            if (memcmp(D.buf->h.data() + i * es, D.buf->h0.data() + i * es, es) != 0) r.untouchedOk = false;
        }
    }
    return r;
}

static void report(const std::string& id, const CheckResult& r, cudaDataType_t dt, const std::string& note, bool expectErr = false);

static void reportTol(const std::string& id, const CheckResult& r, double tol, const std::string& note) {
    if (r.anyErr || !r.haveResult) {
        report(id, r, CUDA_R_64F, note);
        return;
    }
    double norm = r.maxRef > 0 ? r.maxAbs / r.maxRef : r.maxAbs;
    printf("[%s] %s | maxabs=%.3e maxrel=%.3e maxabs/max|ref|=%.3e (tolerance %.0e) | %s\n", id.c_str(),
           norm <= tol ? "ok" : "FAIL", r.maxAbs, r.maxRel, norm, tol, note.c_str());
    fflush(stdout);
}

static void report(const std::string& id, const CheckResult& r, cudaDataType_t dt, const std::string& note, bool expectErr) {
    std::string v, detail;
    if (r.anyErr) {
        v = expectErr ? "ERR (probed)" : "ERR (UNEXPECTED)";
        detail = "stage=" + r.errStage + " " + r.errStr;
    } else if (!r.haveResult) {
        v = "ERR";
        detail = "no result";
    } else {
        double norm = r.maxRef > 0 ? r.maxAbs / r.maxRef : r.maxAbs;
        char buf[256];
        snprintf(buf, sizeof buf, "maxabs=%.3e maxrel=%.3e maxabs/max|ref|=%.3e", r.maxAbs, r.maxRel, norm);
        detail = buf;
        if (r.untouchedChecked) {
            detail += r.untouchedOk ? " | untouched-elems-preserved" : " | UNTOUCHED ELEMENTS MODIFIED";
            if (r.untouchedOk && r.nUntouched == 0) detail += " (none)";
        }
        v = (norm <= tolOf(dt) && r.untouchedOk) ? "ok" : "FAIL";
    }
    printf("[%s] %s | %s | %s\n", id.c_str(), v.c_str(), detail.c_str(), note.c_str());
    fflush(stdout);
}

// Picks the CPU accumulator from the operand types (complex if any operand is complex; 64-bit if any is 64-bit).
static CheckResult runTyped(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, Tsr& A,
                            cutensorOperator_t opA, Tsr& B, cutensorOperator_t opB, Tsr& C, cutensorOperator_t opC, Tsr& D,
                            cutensorComputeDescriptor_t comp, const Scalars& sc, bool checkUntouched, OpHook opHook = nullptr) {
    bool cplx = isCplx(A.s.dt) || isCplx(B.s.dt) || isCplx(D.s.dt);
    bool wide = A.s.dt == CUDA_R_64F || A.s.dt == CUDA_C_64F || B.s.dt == CUDA_R_64F || B.s.dt == CUDA_C_64F ||
                D.s.dt == CUDA_R_64F || D.s.dt == CUDA_C_64F;
    if (cplx && wide) return doCheck<W>(h, stream, ws, wsSize, A, opA, B, opB, C, opC, D, comp, sc, checkUntouched, -1, opHook);
    if (cplx) return doCheck<CF>(h, stream, ws, wsSize, A, opA, B, opB, C, opC, D, comp, sc, checkUntouched, -1, opHook);
    if (wide) return doCheck<double>(h, stream, ws, wsSize, A, opA, B, opB, C, opC, D, comp, sc, checkUntouched, -1, opHook);
    return doCheck<float>(h, stream, ws, wsSize, A, opA, B, opB, C, opC, D, comp, sc, checkUntouched, -1, opHook);
}

static const cutensorOperator_t ID = CUTENSOR_OP_IDENTITY, CJ = CUTENSOR_OP_CONJ;
static const int M0 = 37, K0 = 53, N0 = 41;  // deliberately non-power-of-two GEMM extents
