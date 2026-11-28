// vs.cpp -- cross-version cuTENSOR 2.x probe. Compiled once against the 2.7.0 header and run against every
// libcutensor.so.2 installed on the machine (LD_LIBRARY_PATH); the 2.x ABI is stable across these versions.
// Cases (one line each): S1 stride-0 input, S2 stride-0 reduced mode, S3 stride-0 output, S4 opC=CONJ with beta!=0,
// S5 duplicate mode, S6 over-declared alignment, S7 extent mismatch.  Usage: vs [all | S1 S2 ...]
#include <cutensor.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cx = std::complex<double>;
static const cutensorOperator_t ID = CUTENSOR_OP_IDENTITY;
static const cutensorOperator_t CONJ = CUTENSOR_OP_CONJ;
static const cutensorComputeDescriptor_t COMP = CUTENSOR_COMPUTE_DESC_64F;
static const int NI = 4, NJ = 3, NK = 5;  // tiny extents
static const uint64_t WS_LIMIT = 1ull << 20, WS_MAX = 256ull << 20;
static const cx SENTINEL(-7.5e300, 0);

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

static bool isok(cutensorStatus_t s) { return s == CUTENSOR_STATUS_SUCCESS; }
static std::string f2(double v) { char b[64]; snprintf(b, sizeof b, "%.2e", v); return b; }

// ---------------------------------------------------------------- specs and device buffers
struct Spec {  // layout: modes, extents, strides (elements), dtype, declared alignment
    std::vector<int32_t> modes;
    std::vector<int64_t> ext, str;
    cudaDataType_t dt = CUDA_R_64F;
    uint32_t align = 256;
};
static Spec packed(std::vector<int32_t> m, std::vector<int64_t> e, cudaDataType_t dt = CUDA_R_64F, uint32_t a = 256) {
    std::vector<int64_t> s(e.size());
    int64_t acc = 1;
    for (size_t i = 0; i < e.size(); ++i) { s[i] = acc; acc *= e[i]; }
    return Spec{std::move(m), std::move(e), std::move(s), dt, a};
}

struct Buf {
    cudaDataType_t dt = CUDA_R_64F;
    size_t n = 0, esz = 8;
    int64_t off = 0;  // element offset of the tensor base inside the buffer
    void* d = nullptr;
    std::vector<cx> h;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf(Buf&& o) : dt(o.dt), n(o.n), esz(o.esz), off(o.off), d(o.d), h(std::move(o.h)) { o.d = nullptr; }
    Buf& operator=(Buf&&) = delete;
    ~Buf() { if (d) cudaFree(d); }
    void* p() const { return (char*)d + off * (int64_t)esz; }
};

// allocate n (+pad) elements, tensor base at element off; random uniform [-1,1] (complex parts for c64) or constant
static Buf make(size_t n, cudaDataType_t dt, std::mt19937_64& rng, size_t pad = 0, int64_t off = 0, cx fill = SENTINEL,
                bool random = true) {
    Buf b;
    b.dt = dt; b.esz = dt == CUDA_C_64F ? 16 : 8; b.n = n + pad; b.off = off;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    b.h.assign(b.n, fill);
    if (random)
        for (auto& v : b.h) {
            double re = u(rng); if (std::fabs(re) < 0.05) re += 0.3;
            double im = u(rng); if (std::fabs(im) < 0.05) im -= 0.3;
            v = dt == CUDA_C_64F ? cx(re, im) : cx(re, 0);
        }
    CUDA_CHECK(cudaMalloc(&b.d, b.n * b.esz));
    std::vector<char> raw(b.n * b.esz);
    for (size_t i = 0; i < b.n; ++i) {
        if (dt == CUDA_C_64F) { cuDoubleComplex z = make_cuDoubleComplex(b.h[i].real(), b.h[i].imag()); memcpy(&raw[i * 16], &z, 16); }
        else { double v = b.h[i].real(); memcpy(&raw[i * 8], &v, 8); }
    }
    CUDA_CHECK(cudaMemcpy(b.d, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    return b;
}
static void download(Buf& b) {
    std::vector<char> raw(b.n * b.esz);
    CUDA_CHECK(cudaMemcpy(raw.data(), b.d, raw.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < b.n; ++i) {
        if (b.dt == CUDA_C_64F) { cuDoubleComplex z; memcpy(&z, &raw[i * 16], 16); b.h[i] = cx(z.x, z.y); }
        else { double v; memcpy(&v, &raw[i * 8], 8); b.h[i] = cx(v, 0); }
    }
}

// ---------------------------------------------------------------- status-tracked pipeline
struct Stages {
    cutensorStatus_t desc[4] = {CUTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS,
                                CUTENSOR_STATUS_SUCCESS};  // A, B, C, D
    cutensorStatus_t op = CUTENSOR_STATUS_SUCCESS, pref = CUTENSOR_STATUS_SUCCESS, ws = CUTENSOR_STATUS_SUCCESS,
                     plan = CUTENSOR_STATUS_SUCCESS, exec = CUTENSOR_STATUS_SUCCESS;
    uint64_t wsEst = 0, wsReq = 0;
    std::string opName, cudaErr;
    std::string firstFail() const {
        const char* nm[4] = {"descA", "descB", "descC", "descD"};
        for (int i = 0; i < 4; ++i) if (!isok(desc[i])) return nm[i];
        if (!isok(op)) return opName;
        if (!isok(pref)) return "CreatePlanPreference";
        if (!isok(ws)) return "EstimateWorkspaceSize";
        if (!isok(plan)) return "CreatePlan";
        if (!isok(exec)) return "execute";
        return "";
    }
    cutensorStatus_t firstStatus() const {
        for (int i = 0; i < 4; ++i) if (!isok(desc[i])) return desc[i];
        if (!isok(op)) return op;
        if (!isok(pref)) return pref;
        if (!isok(ws)) return ws;
        if (!isok(plan)) return plan;
        return exec;
    }
    bool ok() const { return firstFail().empty(); }
    std::string err() const {
        std::string s = "ERR(" + firstFail() + " status=" + std::to_string((int)firstStatus()) + " " +
                        cutensorGetErrorString(firstStatus()) + ")";
        if (!cudaErr.empty()) s += " cuda: " + cudaErr;
        return s;
    }
};

// descriptors -> CreateContraction/CreateReduction -> plan pref -> ws estimate -> plan -> execute (default stream)
static void pipeline(cutensorHandle_t h, bool reduce, const Spec* S[4], const cutensorOperator_t ops[3], const void* alpha,
                     const void* beta, void* ptr[4], void* ws, uint64_t wsLimit, Stages& st) {
    cutensorTensorDescriptor_t d[4] = {nullptr, nullptr, nullptr, nullptr};
    cutensorOperationDescriptor_t od = nullptr;
    cutensorPlanPreference_t pref = nullptr;
    cutensorPlan_t plan = nullptr;
    st.opName = reduce ? "CreateReduction" : "CreateContraction";
    for (int i = 0; i < 4; ++i)
        st.desc[i] = cutensorCreateTensorDescriptor(h, &d[i], (uint32_t)S[i]->ext.size(), S[i]->ext.empty() ? nullptr : S[i]->ext.data(),
                                                    S[i]->str.empty() ? nullptr : S[i]->str.data(), S[i]->dt, S[i]->align);
    if (st.ok()) {
        if (reduce)
            st.op = cutensorCreateReduction(h, &od, d[0], S[0]->modes.data(), ops[0], d[2], S[2]->modes.data(), ops[2], d[3],
                                            S[3]->modes.data(), CUTENSOR_OP_ADD, COMP);
        else
            st.op = cutensorCreateContraction(h, &od, d[0], S[0]->modes.data(), ops[0], d[1], S[1]->modes.data(), ops[1], d[2],
                                              S[2]->modes.data(), ops[2], d[3], S[3]->modes.data(), COMP);
    }
    if (st.ok()) st.pref = cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
    if (st.ok()) st.ws = cutensorEstimateWorkspaceSize(h, od, pref, CUTENSOR_WORKSPACE_DEFAULT, &st.wsEst);
    if (st.ok()) st.plan = cutensorCreatePlan(h, &plan, od, pref, wsLimit);
    if (st.ok()) {
        cutensorPlanGetAttribute(h, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &st.wsReq, sizeof(st.wsReq));
        st.exec = reduce ? cutensorReduce(h, plan, alpha, ptr[0], beta, ptr[2], ptr[3], ws, wsLimit, nullptr)
                         : cutensorContract(h, plan, alpha, ptr[0], ptr[1], beta, ptr[2], ptr[3], ws, wsLimit, nullptr);
        if (isok(st.exec)) {
            cudaError_t ce = cudaDeviceSynchronize();
            if (ce != cudaSuccess) { st.exec = CUTENSOR_STATUS_CUDA_ERROR; st.cudaErr = cudaGetErrorString(ce); }
        }
    }
    if (plan) cutensorDestroyPlan(plan);
    if (od) cutensorDestroyOperationDescriptor(od);
    if (pref) cutensorDestroyPlanPreference(pref);
    for (int i = 0; i < 4; ++i) if (d[i]) cutensorDestroyTensorDescriptor(d[i]);
}

// 1 MiB workspace limit; if planning fails, retry once with 256 MiB (noted in the returned suffix)
static std::string run(cutensorHandle_t h, bool reduce, const Spec* S[4], const cutensorOperator_t ops[3], const void* alpha,
                       const void* beta, void* ptr[4], void* ws, Stages& st) {
    pipeline(h, reduce, S, ops, alpha, beta, ptr, ws, WS_LIMIT, st);
    if (!isok(st.plan) && isok(st.ws)) {
        Stages st2;
        pipeline(h, reduce, S, ops, alpha, beta, ptr, ws, WS_MAX, st2);
        if (st2.ok()) { st = st2; return " [plan needed ws > 1 MiB, retried with 256 MiB]"; }
    }
    return "";
}

// ---------------------------------------------------------------- comparison
struct Cmp { double err = 0, scale = 0; bool close = false; };
static Cmp compareVec(const std::vector<cx>& got, const std::vector<cx>& ref) {
    Cmp c;
    for (size_t i = 0; i < ref.size(); ++i) { c.err = std::max(c.err, std::abs(got[i] - ref[i])); c.scale = std::max(c.scale, std::abs(ref[i])); }
    c.close = c.err <= 1e-12 * std::max(1.0, c.scale);
    return c;
}
static Cmp compare(const Buf& d, const std::vector<cx>& ref) { return compareVec(d.h, ref); }
static std::string okWrong(const Stages& st, const Buf& d, const std::vector<cx>& ref) {
    if (!st.ok()) return st.err();
    Cmp c = compare(d, ref);
    return c.close ? "ok" : "WRONG(maxabs " + f2(c.err) + ")";
}

// ---------------------------------------------------------------- cases
static const cutensorOperator_t OPS_ID[3] = {ID, ID, ID};

// S1: ij,jk->ik f64, A = [i,j] ext (4,j) strides (0,1): a j-vector broadcast over i
static void caseS1(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    Spec A{{'i', 'j'}, {NI, NJ}, {0, 1}};
    Spec B = packed({'j', 'k'}, {NJ, NK}), C = packed({'i', 'k'}, {NI, NK}), D = C;
    Buf a = make(NJ, CUDA_R_64F, rng), b = make(NJ * NK, CUDA_R_64F, rng), c = make(NI * NK, CUDA_R_64F, rng),
        d = make(NI * NK, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S[4] = {&A, &B, &C, &D};
    double alpha = 1, beta = 0;
    void* ptr[4] = {a.p(), b.p(), c.p(), d.p()};
    Stages st;
    std::string note = run(h, false, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    download(d);
    std::vector<cx> ref(NI * NK);
    for (int i = 0; i < NI; ++i) for (int k = 0; k < NK; ++k) { cx s = 0; for (int j = 0; j < NJ; ++j) s += a.h[j] * b.h[j + (size_t)k * NJ]; ref[i + (size_t)k * NI] = s; }
    Cmp cm = compare(d, ref);
    printf("S1 | %s | ij,jk->ik f64, A modes[i,j] ext(4,%d) strides(0,1) (j-vector broadcast over i), alpha=1 beta=0 | maxabs %s, ref scale %.2f, wsEst %llu wsReq %llu%s\n",
           okWrong(st, d, ref).c_str(), NJ, f2(cm.err).c_str(), cm.scale, (unsigned long long)st.wsEst, (unsigned long long)st.wsReq, note.c_str());
}

// S2: ij->j f64 with A stride 0 on the reduced mode i (ext 4): expected D = 4*A[0,j]; via the reduction API and via ij,j->j (B=1)
static void caseS2(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    Spec A{{'i', 'j'}, {NI, NJ}, {0, 1}};
    Spec Bs = packed({'j'}, {NJ}), Cj = packed({'j'}, {NJ}), Dj = Cj;
    Buf a = make(NJ, CUDA_R_64F, rng), b = make(NJ, CUDA_R_64F, rng, 0, 0, cx(1, 0), false), c = make(NJ, CUDA_R_64F, rng),
        d = make(NJ, CUDA_R_64F, rng, 0, 0, SENTINEL, false), d2 = make(NJ, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    std::vector<cx> ref(NJ);
    for (int j = 0; j < NJ; ++j) ref[j] = (double)NI * a.h[j];
    double alpha = 1, beta = 0;
    const Spec* S[4] = {&A, &A, &Cj, &Dj};  // reduction: B unused
    void* ptr[4] = {a.p(), nullptr, c.p(), d.p()};
    Stages st;
    std::string n1 = run(h, true, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    download(d);
    const Spec* S2[4] = {&A, &Bs, &Cj, &Dj};
    void* ptr2[4] = {a.p(), b.p(), c.p(), d2.p()};
    Stages st2;
    std::string n2 = run(h, false, S2, OPS_ID, &alpha, &beta, ptr2, ws, st2);
    download(d2);
    printf("S2 | reduce-api ij->j (A ext(4,%d) strides(0,1), ADD): %s | contract-api ij,j->j (B=1): %s | expected D = 4*A[0,j]%s%s\n",
           NJ, okWrong(st, d, ref).c_str(), okWrong(st2, d2, ref).c_str(), n1.c_str(), n2.c_str());
}

// S3: ij,jk->ik with a stride-0 output. Variants isolate the factors behind acceptance: D layout (0,1)/(1,0), whether C
// carries the same stride-0 layout (yesterday's 2c did), and size (tiny vs 37x53x41 = yesterday's base GEMM shape).
static std::string s3variant(cutensorHandle_t h, void* ws, std::mt19937_64& rng, int ni, int nj, int nk, std::vector<int64_t> strides,
                             bool cSame) {
    Spec A = packed({'i', 'j'}, {ni, nj}), B = packed({'j', 'k'}, {nj, nk});
    Spec C = cSame ? Spec{{'i', 'k'}, {ni, nk}, strides} : packed({'i', 'k'}, {ni, nk});
    Spec D{{'i', 'k'}, {ni, nk}, strides};
    size_t bufLen = (size_t)(strides[0] * (ni - 1) + strides[1] * (nk - 1) + 1);
    Buf a = make(ni * nj, CUDA_R_64F, rng), b = make(nj * nk, CUDA_R_64F, rng), c = make(cSame ? bufLen : ni * nk, CUDA_R_64F, rng),
        d = make(bufLen, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S[4] = {&A, &B, &C, &D};
    double alpha = 1, beta = 0;
    void* ptr[4] = {a.p(), b.p(), c.p(), d.p()};
    Stages st;
    std::string note = run(h, false, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    if (!st.ok()) return st.err() + note;
    download(d);
    int best = -1; double bestErr = 1e300;
    for (int i = 0; i < ni; ++i) {  // which reference row (if any) did the buffer end up holding?
        double e = 0;
        for (int k = 0; k < nk; ++k) {
            cx s = 0;
            for (int j = 0; j < nj; ++j) s += a.h[i + (size_t)j * ni] * b.h[j + (size_t)k * nj];
            e = std::max(e, std::abs(d.h[strides[0] * i + strides[1] * k] - s));
        }
        if (e < bestErr) { bestErr = e; best = i; }
    }
    return (bestErr <= 1e-12 ? "accepted: buffer holds reference row i=" + std::to_string(best) + " (one writer won)"
                             : "accepted: garbage (closest reference row i=" + std::to_string(best) + ", maxabs " + f2(bestErr) + ")") + note;
}
static void caseS3(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    std::string a = s3variant(h, ws, rng, NI, NJ, NK, {0, 1}, false);
    std::string b = s3variant(h, ws, rng, NI, NJ, NK, {1, 0}, false);
    std::string c = s3variant(h, ws, rng, NI, NJ, NK, {1, 0}, true);
    std::string d = s3variant(h, ws, rng, 37, 53, 41, {1, 0}, true);
    std::string e = s3variant(h, ws, rng, NI, NJ, NK, {0, 1}, true);
    printf("S3 | ij,jk->ik f64, stride-0 output | (a) 4x3x5 D strides(0,1), C packed: %s | (b) 4x3x5 D strides(1,0), C packed: %s | "
           "(c) 4x3x5 D strides(1,0), C same layout: %s | (d) 37x53x41 D strides(1,0), C same layout (=old 2c): %s | "
           "(e) 4x3x5 D strides(0,1), C same layout: %s\n",
           a.c_str(), b.c_str(), c.c_str(), d.c_str(), e.c_str());
}

// S4: ij,jk->ik c64, opC=CONJ, beta=0.3, C aliases D (pre-filled): "ignored" vs "applied"
static void caseS4(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    Spec A = packed({'i', 'j'}, {NI, NJ}, CUDA_C_64F), B = packed({'j', 'k'}, {NJ, NK}, CUDA_C_64F),
         C = packed({'i', 'k'}, {NI, NK}, CUDA_C_64F), D = C;
    Buf a = make(NI * NJ, CUDA_C_64F, rng), b = make(NJ * NK, CUDA_C_64F, rng), d = make(NI * NK, CUDA_C_64F, rng);
    std::vector<cx> c0 = d.h, refIgn(NI * NK), refConj(NI * NK);
    for (int i = 0; i < NI; ++i) for (int k = 0; k < NK; ++k) { cx s = 0; for (int j = 0; j < NJ; ++j) s += a.h[i + (size_t)j * NI] * b.h[j + (size_t)k * NJ]; size_t id = i + (size_t)k * NI; refIgn[id] = s + 0.3 * c0[id]; refConj[id] = s + 0.3 * std::conj(c0[id]); }
    const cutensorOperator_t ops[3] = {ID, ID, CONJ};
    const Spec* S[4] = {&A, &B, &C, &D};
    cuDoubleComplex alpha = make_cuDoubleComplex(1, 0), beta = make_cuDoubleComplex(0.3, 0);
    void* ptr[4] = {a.p(), b.p(), d.p(), d.p()};
    Stages st;
    std::string note = run(h, false, S, ops, &alpha, &beta, ptr, ws, st);
    std::string v, diffs = "-";
    if (!st.ok()) v = st.err();
    else {
        download(d);
        Cmp ci = compare(d, refIgn), cc = compare(d, refConj);
        v = ci.close ? "ignored" : cc.close ? "applied" : "WRONG(maxabs vs plain-C ref " + f2(ci.err) + ", vs conj ref " + f2(cc.err) + ")";
        diffs = f2(ci.err) + ", conj ref diff " + f2(cc.err);
    }
    printf("S4 | %s | ij,jk->ik c64, opC=CONJ, alpha=1 beta=0.3, C aliases pre-filled D | plain-C ref diff %s%s\n", v.c_str(),
           diffs.c_str(), note.c_str());
}

// S5: A modes [i,i] ext (4,4) strides (4,1) (row-major 4x4): expected D[i,k] = A[i,i]*B[i,k] (sum-of-strides diagonal)
static void caseS5(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    Spec A{{'i', 'i'}, {NI, NI}, {NI, 1}};
    Spec B = packed({'i', 'k'}, {NI, NK}), C = packed({'i', 'k'}, {NI, NK}), D = C;
    Buf a = make(NI * NI, CUDA_R_64F, rng), b = make(NI * NK, CUDA_R_64F, rng), c = make(NI * NK, CUDA_R_64F, rng),
        d = make(NI * NK, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S[4] = {&A, &B, &C, &D};
    double alpha = 1, beta = 0;
    void* ptr[4] = {a.p(), b.p(), c.p(), d.p()};
    Stages st;
    std::string note = run(h, false, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    std::vector<cx> refD(NI * NK), ref1(NI * NK), ref2(NI * NK);
    for (int i = 0; i < NI; ++i) for (int k = 0; k < NK; ++k) {
        cx bk = b.h[i + (size_t)k * NI];
        size_t id = i + (size_t)k * NI;
        refD[id] = a.h[i * NI + i] * bk;  // diagonal: offset i*(NI+1)
        ref1[id] = a.h[i * NI] * bk;      // first occurrence only: offset i*NI
        ref2[id] = a.h[i] * bk;           // second occurrence only: offset i
    }
    std::string v = st.err();
    if (st.ok()) {
        download(d);
        Cmp cd = compare(d, refD), c1 = compare(d, ref1), c2 = compare(d, ref2);
        v = cd.close ? "ok-diagonal"
                     : "WRONG(maxabs vs diagonal " + f2(cd.err) + (c1.close ? "; matches FIRST occurrence only (offset 4i)" : c2.close ? "; matches SECOND occurrence only (offset i)" : "; matches none of diag/first/second") + ")";
    }
    printf("S5 | %s | A modes[i,i] ext(4,4) strides(4,1) row-major, ii,ik->ik, expected D[i,k]=A[i,i]*B[i,k]%s\n", v.c_str(), note.c_str());
}

// S6: A base = cudaMalloc + 1 element (8 B), alignmentRequirement declared 256
static void caseS6(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    Spec A = packed({'i', 'j'}, {NI, NJ}, CUDA_R_64F, 256);
    Spec B = packed({'j', 'k'}, {NJ, NK}), C = packed({'i', 'k'}, {NI, NK}), D = C;
    Buf a = make(NI * NJ, CUDA_R_64F, rng, 1, 1), b = make(NJ * NK, CUDA_R_64F, rng), c = make(NI * NK, CUDA_R_64F, rng),
        d = make(NI * NK, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S[4] = {&A, &B, &C, &D};
    double alpha = 1, beta = 0;
    void* ptr[4] = {a.p(), b.p(), c.p(), d.p()};
    Stages st;
    std::string note = run(h, false, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    download(d);
    std::vector<cx> ref(NI * NK);
    for (int i = 0; i < NI; ++i) for (int k = 0; k < NK; ++k) { cx s = 0; for (int j = 0; j < NJ; ++j) s += a.h[1 + i + (size_t)j * NI] * b.h[j + (size_t)k * NJ]; ref[i + (size_t)k * NI] = s; }
    printf("S6 | %s | ij,jk->ik f64, A ptr = cudaMalloc + 8 B (actual alignment %zu B), alignmentRequirement=256%s\n",
           okWrong(st, d, ref).c_str(), (size_t)((uintptr_t)a.p() % 256), note.c_str());
}

// S7: extent mismatch: (a) contracted mode j: A=3, B=4; (b) batch mode k: B=3, D=4
static void caseS7(cutensorHandle_t h, void* ws, std::mt19937_64& rng) {
    double alpha = 1, beta = 0;
    // (a)
    Spec A{{'i', 'j'}, {NI, 3}, {1, NI}};  // ld = NI, 3 columns described, 4 allocated
    Spec B = packed({'j', 'k'}, {4, NK}), C = packed({'i', 'k'}, {NI, NK}), D = C;
    Buf a = make(NI * 4, CUDA_R_64F, rng), b = make(4 * NK, CUDA_R_64F, rng), c = make(NI * NK, CUDA_R_64F, rng),
        d = make(NI * NK, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S[4] = {&A, &B, &C, &D};
    void* ptr[4] = {a.p(), b.p(), c.p(), d.p()};
    Stages st;
    std::string n1 = run(h, false, S, OPS_ID, &alpha, &beta, ptr, ws, st);
    std::vector<cx> ref3(NI * NK), ref4(NI * NK);
    for (int i = 0; i < NI; ++i) for (int k = 0; k < NK; ++k) {
        cx s3 = 0, s4 = 0;
        for (int j = 0; j < 4; ++j) { cx t = a.h[i + (size_t)j * NI] * b.h[j + (size_t)k * 4]; s4 += t; if (j < 3) s3 += t; }
        ref3[i + (size_t)k * NI] = s3; ref4[i + (size_t)k * NI] = s4;
    }
    std::string va;
    if (!st.ok()) va = st.err();
    else {
        download(d);
        Cmp c3 = compare(d, ref3), c4 = compare(d, ref4);
        va = "ACCEPTED(all stages): " + std::string(c3.close ? "result = sum over j<3 (A's extent)" : c4.close ? "result = sum over j<4 (B's extent, reads A's 4th column)" : "result wrong (maxabs vs j<3 " + f2(c3.err) + ", vs j<4 " + f2(c4.err) + ")");
    }
    // (b)
    Spec A2 = packed({'i', 'j'}, {NI, NJ});
    Spec B2{{'j', 'k'}, {NJ, 3}, {1, NJ}};  // ld = NJ, 3 columns described, 4 allocated
    Spec C2 = packed({'i', 'k'}, {NI, 4}), D2 = C2;
    Buf b2 = make(NJ * 4, CUDA_R_64F, rng), c2 = make(NI * 4, CUDA_R_64F, rng), d2 = make(NI * 4, CUDA_R_64F, rng, 0, 0, SENTINEL, false);
    const Spec* S2[4] = {&A2, &B2, &C2, &D2};
    void* ptr2[4] = {a.p(), b2.p(), c2.p(), d2.p()};
    Stages st2;
    std::string n2 = run(h, false, S2, OPS_ID, &alpha, &beta, ptr2, ws, st2);
    std::string vb;
    if (!st2.ok()) vb = st2.err();
    else {
        download(d2);
        std::vector<cx> ref(NI * 4);
        for (int i = 0; i < NI; ++i) for (int k = 0; k < 4; ++k) { cx s = 0; for (int j = 0; j < NJ; ++j) s += a.h[i + (size_t)j * NI] * b2.h[j + (size_t)k * NJ]; ref[i + (size_t)k * 4] = s; }
        // D2 is column-major (ld NI): elements 0..NI*3 are columns k<3; column k=3 starts at element 3*NI
        std::vector<cx> common(NI * 3), gotCommon(d2.h.begin(), d2.h.begin() + NI * 3);
        for (int i = 0; i < NI; ++i) for (int k = 0; k < 3; ++k) common[i + (size_t)k * NI] = ref[i + (size_t)k * 4];
        double e4 = 0, z4 = 0;
        for (int i = 0; i < NI; ++i) { e4 = std::max(e4, std::abs(d2.h[i + 3 * NI] - ref[i + 3 * 4])); z4 = std::max(z4, std::abs(d2.h[i + 3 * NI])); }
        Cmp cc = compareVec(gotCommon, common);
        vb = "ACCEPTED(all stages): k<3 " + std::string(cc.close ? "ok" : "WRONG(maxabs " + f2(cc.err) + ")") + "; D[:,3] = " +
              (z4 == 0 ? "0" : e4 <= 1e-12 ? "B's 4th allocated column (read past B's described extent)" : "garbage (maxabs vs B col 3 " + f2(e4) + ")");
    }
    printf("S7 | contracted-mode mismatch (j: A=3, B=4): %s | batch-mode mismatch (k: B=3, D=4): %s%s%s\n", va.c_str(), vb.c_str(), n1.c_str(), n2.c_str());
}

// ---------------------------------------------------------------- driver
static void printVersion(cutensorHandle_t h) {
    size_t v = cutensorGetVersion(), cv = cutensorGetCudartVersion();
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("cutensorGetVersion() = %zu (%zu.%zu.%zu) | cutensorGetCudartVersion() = %zu (%zu.%zu) | cudart in binary %d, driver %d | GPU \"%s\" sm_%d%d\n",
           v, v / 10000, (v / 100) % 100, v % 100, cv, cv / 1000, (cv / 10) % 100, rt, drv, p.name, p.major, p.minor);
    fflush(stdout);
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    printf("cutensorGetVersion() = %zu (%zu.%zu.%zu)\n", cutensorGetVersion(), cutensorGetVersion() / 10000,
           (cutensorGetVersion() / 100) % 100, cutensorGetVersion() % 100);  // needs no handle: printed even if creation fails
    cutensorHandle_t h = nullptr;
    cutensorStatus_t s = cutensorCreate(&h);
    if (!isok(s)) { printf("cutensorCreate failed: status %d %s\n", (int)s, cutensorGetErrorString(s)); return 1; }
    printVersion(h);
    void* ws = nullptr;
    CUDA_CHECK(cudaMalloc(&ws, WS_MAX));
    std::vector<std::string> sel(argv + 1, argv + argc);
    bool all = sel.empty() || (sel.size() == 1 && sel[0] == "all");
    struct Case { const char* id; void (*fn)(cutensorHandle_t, void*, std::mt19937_64&); } cases[] = {
        {"S1", caseS1}, {"S2", caseS2}, {"S3", caseS3}, {"S4", caseS4}, {"S5", caseS5}, {"S6", caseS6}, {"S7", caseS7}};
    for (auto& c : cases)
        if (all || std::find(sel.begin(), sel.end(), c.id) != sel.end()) {
            std::mt19937_64 rng(20261128);  // identical data in combined and isolated runs
            c.fn(h, ws, rng);
            fflush(stdout);
        }
    cudaFree(ws);
    cutensorDestroy(h);
    printf("--- done ---\n");
    return 0;
}
