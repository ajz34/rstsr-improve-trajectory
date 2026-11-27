// Checks 20-23: host planning cost, device time vs cuBLAS, non-GEMM-shaped contraction, many-mode contraction.
#include "probe_common.h"

#include <numeric>

static double msSince(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------
// Check 20: planning cost (fresh descriptors every time)
struct PlanTiming {
    double tDesc = 0, tOp = 0, tPref = 0, tWs = 0, tPlan = 0;
    cutensorStatus_t st = CUTENSOR_STATUS_SUCCESS;
    uint64_t wsEst = 0, wsReq = 0;
    std::string failStage;
};

static PlanTiming planOnce(cutensorHandle_t h, int m, int k, int n, cudaDataType_t dt, cutensorComputeDescriptor_t comp,
                          int cacheMode, uint64_t wsLimit) {
    PlanTiming t;
    TSpec sA = colMajor({'m', 'k'}, {m, k}, dt), sB = colMajor({'k', 'n'}, {k, n}, dt), sD = colMajor({'m', 'n'}, {m, n}, dt);
    cutensorTensorDescriptor_t dA = nullptr, dB = nullptr, dD = nullptr;
    cutensorOperationDescriptor_t op = nullptr;
    cutensorPlanPreference_t pref = nullptr;
    cutensorPlan_t plan = nullptr;
    auto t0 = clk::now();
    cutensorStatus_t s1 = cutensorCreateTensorDescriptor(h, &dA, 2, sA.extent.data(), sA.stride.data(), dt, 256);
    cutensorStatus_t s2 = cutensorCreateTensorDescriptor(h, &dB, 2, sB.extent.data(), sB.stride.data(), dt, 256);
    cutensorStatus_t s3 = cutensorCreateTensorDescriptor(h, &dD, 2, sD.extent.data(), sD.stride.data(), dt, 256);
    auto t1 = clk::now();
    cutensorStatus_t s4 = (isok(s1) && isok(s2) && isok(s3))
                             ? cutensorCreateContraction(h, &op, dA, sA.modes.data(), ID, dB, sB.modes.data(), ID, dD,
                                                         sD.modes.data(), ID, dD, sD.modes.data(), comp)
                             : (isok(s1) ? s2 : s1);
    auto t2 = clk::now();
    cutensorStatus_t s5 = isok(s4) ? cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE) : s4;
    if (isok(s5) && cacheMode >= 0) {
        int32_t cm = cacheMode;
        cutensorPlanPreferenceSetAttribute(h, pref, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cm, sizeof(cm));
    }
    auto t3 = clk::now();
    cutensorStatus_t s6 = isok(s5) ? cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &t.wsEst) : s5;
    auto t4 = clk::now();
    cutensorStatus_t s7 = isok(s6) ? cutensorCreatePlan(h, &plan, op, pref, wsLimit) : s6;
    auto t5 = clk::now();
    if (isok(s7)) cutensorPlanGetAttribute(h, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &t.wsReq, sizeof(t.wsReq));
    t.tDesc = msSince(t0, t1);
    t.tOp = msSince(t1, t2);
    t.tPref = msSince(t2, t3);
    t.tWs = msSince(t3, t4);
    t.tPlan = msSince(t4, t5);
    t.st = s7;
    if (!isok(s7)) {
        const char* stage = !isok(s1) ? "descA" : !isok(s2) ? "descB" : !isok(s3) ? "descD" : !isok(s4) ? "CreateContraction"
                                   : !isok(s5)                          ? "CreatePlanPreference"
                                   : !isok(s6)                          ? "EstimateWorkspaceSize"
                                                                        : "CreatePlan";
        t.failStage = stage;
    }
    if (plan) cutensorDestroyPlan(plan);
    if (pref) cutensorDestroyPlanPreference(pref);
    if (op) cutensorDestroyOperationDescriptor(op);
    if (dA) cutensorDestroyTensorDescriptor(dA);
    if (dB) cutensorDestroyTensorDescriptor(dB);
    if (dD) cutensorDestroyTensorDescriptor(dD);
    return t;
}

static void printPlanTiming(const char* what, const PlanTiming& t) {
    if (!isok(t.st)) {
        printf("[20] %s | ERR | stage=%s status=%d (%s)\n", what, t.failStage.c_str(), (int)t.st, sname(t.st));
        return;
    }
    printf("[20] %s | ok | total=%.3f ms (CreateTensorDescriptor x3 %.3f, CreateContraction %.3f, CreatePlanPreference %.3f, "
           "EstimateWorkspaceSize %.3f, CreatePlan %.3f) | wsEstimate=%llu B wsRequired=%llu B\n",
           what, t.tDesc + t.tOp + t.tPref + t.tWs + t.tPlan, t.tDesc, t.tOp, t.tPref, t.tWs, t.tPlan,
           (unsigned long long)t.wsEst, (unsigned long long)t.wsReq);
    fflush(stdout);
}

static void checkPlanning(cutensorHandle_t h) {
    printf("--- check 20: host-side planning cost (f64 GEMM-shaped, CUTENSOR_ALGO_DEFAULT, JIT none, fresh process) ---\n");
    const uint64_t WS = 128ull << 20;
    PlanTiming t;
    // (0) very first planning call in the process: includes one-time library initialisation
    t = planOnce(h, 8, 8, 8, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
    printPlanTiming("first-ever plan in process, 8^3 (includes library init)", t);
    // (a) first time for the 2048^3 problem
    t = planOnce(h, 2048, 2048, 2048, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
    printPlanTiming("2048^3 FIRST (library initialised, problem new)", t);
    // (b) repeated identical problem x20
    {
        const int R = 20;
        std::vector<double> tot(R), plan(R), op(R), ws(R);
        for (int i = 0; i < R; ++i) {
            PlanTiming p = planOnce(h, 2048, 2048, 2048, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
            tot[i] = p.tDesc + p.tOp + p.tPref + p.tWs + p.tPlan;
            plan[i] = p.tPlan;
            op[i] = p.tOp;
            ws[i] = p.tWs;
        }
        std::sort(tot.begin(), tot.end());
        std::sort(plan.begin(), plan.end());
        double mean = std::accumulate(tot.begin(), tot.end(), 0.0) / R;
        printf("[20] 2048^3 REPEATED x%d (identical problem; default plan cache) | ok | total median=%.3f ms mean=%.3f ms min=%.3f max=%.3f "
               "| CreatePlan median=%.3f ms | CreateContraction median=%.3f ms | EstimateWorkspaceSize median=%.3f ms\n",
               R, tot[R / 2], mean, tot[0], tot[R - 1], plan[R / 2], op[R / 2], ws[R / 2]);
    }
    // 64^3
    t = planOnce(h, 64, 64, 64, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
    printPlanTiming("64^3 FIRST (library initialised, problem new)", t);
    {
        const int R = 20;
        std::vector<double> tot(R), plan(R), op(R), ws(R);
        for (int i = 0; i < R; ++i) {
            PlanTiming p = planOnce(h, 64, 64, 64, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
            tot[i] = p.tDesc + p.tOp + p.tPref + p.tWs + p.tPlan;
            plan[i] = p.tPlan;
            op[i] = p.tOp;
            ws[i] = p.tWs;
        }
        std::sort(tot.begin(), tot.end());
        std::sort(plan.begin(), plan.end());
        double mean = std::accumulate(tot.begin(), tot.end(), 0.0) / R;
        printf("[20] 64^3 REPEATED x%d (identical problem; default plan cache) | ok | total median=%.3f ms mean=%.3f ms min=%.3f max=%.3f "
               "| CreatePlan median=%.3f ms | CreateContraction median=%.3f ms | EstimateWorkspaceSize median=%.3f ms\n",
               R, tot[R / 2], mean, tot[0], tot[R - 1], plan[R / 2], op[R / 2], ws[R / 2]);
    }
    // a second distinct 2048-class problem (different extents) planned for the first time: pure per-problem cost
    t = planOnce(h, 2050, 2050, 2050, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
    printPlanTiming("2050^3 FIRST (library warm; distinct problem, no cache hit)", t);
    // same 2048^3 problem with the plan cache disabled on the preference
    {
        const int R = 5;
        std::vector<double> tot(R);
        cutensorStatus_t st = CUTENSOR_STATUS_SUCCESS;
        for (int i = 0; i < R; ++i) {
            PlanTiming p = planOnce(h, 2048, 2048, 2048, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_CACHE_MODE_NONE, WS);
            tot[i] = p.tDesc + p.tOp + p.tPref + p.tWs + p.tPlan;
            st = p.st;
        }
        std::sort(tot.begin(), tot.end());
        printf("[20] 2048^3 REPEATED x%d with CUTENSOR_PLAN_PREFERENCE_CACHE_MODE=CUTENSOR_CACHE_MODE_NONE | %s | total median=%.3f ms min=%.3f max=%.3f\n",
               R, isok(st) ? "ok" : "ERR", tot[R / 2], tot[0], tot[R - 1]);
    }
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// device timing
struct Timing {
    double median = 0, mean = 0, min = 0, max = 0;
};
static Timing timeLaunch(const std::function<void()>& launch, int warm, int iters, cudaStream_t stream) {
    for (int i = 0; i < warm; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fflush(stdout);
        fprintf(stderr, "CUDA error during warm-up: %s\n", cudaGetErrorString(ce));
        exit(2);
    }
    std::vector<cudaEvent_t> ev(iters + 1);
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(ev[0], stream));
    for (int i = 0; i < iters; ++i) {
        launch();
        CUDA_CHECK(cudaEventRecord(ev[i + 1], stream));
    }
    CUDA_CHECK(cudaEventSynchronize(ev[iters]));
    std::vector<double> ms(iters);
    for (int i = 0; i < iters; ++i) {
        float f = 0;
        CUDA_CHECK(cudaEventElapsedTime(&f, ev[i], ev[i + 1]));
        ms[i] = f;
    }
    for (auto& e : ev) cudaEventDestroy(e);
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.median = ms[iters / 2];
    t.min = ms[0];
    t.max = ms[iters - 1];
    t.mean = std::accumulate(ms.begin(), ms.end(), 0.0) / iters;
    return t;
}

static void cublasGemm(cublasHandle_t cb, cudaDataType_t dt, int N, const void* A, const void* B, void* D) {
    if (dt == CUDA_R_64F) {
        double one = 1.0, zero = 0.0;
        cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, (const double*)A, N, (const double*)B, N, &zero, (double*)D, N);
    } else if (dt == CUDA_R_32F) {
        float one = 1.0f, zero = 0.0f;
        cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, (const float*)A, N, (const float*)B, N, &zero, (float*)D, N);
    } else if (dt == CUDA_C_64F) {
        cuDoubleComplex one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
        cublasZgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, (const cuDoubleComplex*)A, N, (const cuDoubleComplex*)B, N, &zero,
                    (cuDoubleComplex*)D, N);
    } else if (dt == CUDA_C_32F) {
        cuFloatComplex one = make_cuFloatComplex(1, 0), zero = make_cuFloatComplex(0, 0);
        cublasCgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, (const cuFloatComplex*)A, N, (const cuFloatComplex*)B, N, &zero,
                    (cuFloatComplex*)D, N);
    }
}

// Check 21 for one dtype: cuTENSOR (128 MiB workspace) vs cuBLAS on the same column-major D[m,n]=sum_k A[m,k]B[k,n].
static void gemmPerf(cutensorHandle_t h, cublasHandle_t cb, cudaStream_t stream, void* ws, uint64_t wsSize, cudaDataType_t dt,
                     cutensorComputeDescriptor_t comp, int N, const char* label, std::mt19937_64& rng, int warm, int iters) {
    size_t NN = (size_t)N * N;
    Tsr A(colMajor({'m', 'k'}, {N, N}, dt), rng), B(colMajor({'k', 'n'}, {N, N}, dt), rng), D(colMajor({'m', 'n'}, {N, N}, dt), rng),
        D2(colMajor({'m', 'n'}, {N, N}, dt), rng);
    Scalars sc;
    if (dt == CUDA_R_64F) sc = scalarsF64(1.0, 0.0);
    else if (dt == CUDA_R_32F) sc = scalarsF32(1.0f, 0.0f);
    else if (dt == CUDA_C_64F) sc = scalarsC64(W(1, 0), W(0, 0));
    else sc = scalarsC32(W(1, 0), W(0, 0));
    PlanKeep keep;
    Stages st;
    runContraction(h, A, ID, B, ID, D, ID, D, comp, sc.alpha, sc.beta, ws, wsSize, stream, st, &keep, -1, false);
    if (!st.ok()) {
        printf("[21] %s | ERR | stage=%s status=%d (%s)\n", label, st.firstFail().c_str(), (int)st.firstFailStatus(),
               sname(st.firstFailStatus()));
        fflush(stdout);
        return;
    }
    double flops = (isCplx(dt) ? 8.0 : 2.0) * (double)N * N * N;
    auto cutLaunch = [&]() {
        cutensorContract(h, keep.plan, sc.alpha, A.buf->dptr(), B.buf->dptr(), sc.beta, D.buf->dptr(), D.buf->dptr(), ws, wsSize, stream);
    };
    auto cubLaunch = [&]() { cublasGemm(cb, dt, N, A.buf->d, B.buf->d, D2.buf->d); };
    printf("[21] %s | wsEstimate(DEFAULT)=%llu B wsRequired=%llu B | flops=%.3f G\n", label, (unsigned long long)st.wsEstimate,
           (unsigned long long)st.wsRequired, flops / 1e9);
    for (int round = 1; round <= 2; ++round) {
        Timing tc = timeLaunch(cutLaunch, warm, iters, stream);
        Timing tb = timeLaunch(cubLaunch, warm, iters, stream);
        printf("[21] %s round %d | cuTENSOR median %.3f ms (%.3f TFLOP/s; min %.3f max %.3f) | cuBLAS %s median %.3f ms (%.3f TFLOP/s; min %.3f max %.3f) "
               "| ratio cuBLAS-time/cuTENSOR-time = %.3f\n",
               label, round, tc.median, flops / tc.median / 1e9, tc.min, tc.max, dtName(dt), tb.median, flops / tb.median / 1e9, tb.min,
               tb.max, tb.median / tc.median);
    }
    // cross-check the two results
    D.download();
    D2.download();
    double maxDiff = 0, maxRef = 0;
    for (size_t i = 0; i < NN; ++i) {
        W a = readElem(D.buf->h.data(), dt, (int64_t)i), b = readElem(D2.buf->h.data(), dt, (int64_t)i);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        maxRef = std::max(maxRef, std::abs(b));
    }
    printf("[21] %s cross-check | max|cuTENSOR - cuBLAS| = %.3e, max|cuBLAS| = %.3e, rel = %.3e\n", label, maxDiff, maxRef,
           maxDiff / maxRef);
    fflush(stdout);
}

static void checkGemmPerf(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng) {
    printf("--- check 21: device time (cudaEvent), steady state after warm-up, cuTENSOR vs cuBLAS, same stream ---\n");
    cublasHandle_t cb = nullptr, cb128 = nullptr;
    cublasCreate(&cb);
    cublasSetStream(cb, stream);
    cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH);  // FP32 without TF32 for Sgemm
    int cublasVer = 0;
    cublasGetVersion(cb, &cublasVer);
    printf("[21] cuBLAS version %d, math mode CUBLAS_DEFAULT_MATH (no TF32), default cuBLAS workspace\n", cublasVer);
    const int iters = 10, warm = 3;
    gemmPerf(h, cb, stream, ws, wsSize, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, 2048, "f64 2048^3 (COMPUTE_DESC_64F)", rng, warm, iters);
    gemmPerf(h, cb, stream, ws, wsSize, CUDA_R_32F, CUTENSOR_COMPUTE_DESC_32F, 2048, "f32 2048^3 (COMPUTE_DESC_32F)", rng, warm, iters);
    gemmPerf(h, cb, stream, ws, wsSize, CUDA_C_64F, CUTENSOR_COMPUTE_DESC_64F, 1024, "c64 1024^3 (COMPUTE_DESC_64F)", rng, warm, iters);
    // cuBLAS with an explicit 128 MiB workspace (parity with the cuTENSOR workspace budget)
    {
        void* ws2 = nullptr;
        CUDA_CHECK(cudaMalloc(&ws2, wsSize));
        cublasCreate(&cb128);
        cublasSetStream(cb128, stream);
        cublasSetMathMode(cb128, CUBLAS_DEFAULT_MATH);
        cublasStatus_t cs = cublasSetWorkspace(cb128, ws2, wsSize);
        printf("[21] cuBLAS with explicit %llu MiB workspace (cublasSetWorkspace status %d) ---\n", (unsigned long long)(wsSize >> 20), (int)cs);
        struct C { cudaDataType_t dt; int N; const char* label; };
        const C cases[3] = {{CUDA_R_64F, 2048, "f64 2048^3"}, {CUDA_R_32F, 2048, "f32 2048^3"}, {CUDA_C_64F, 1024, "c64 1024^3"}};
        for (const C& c : cases) {
            Tsr A(colMajor({'m', 'k'}, {c.N, c.N}, c.dt), rng), B(colMajor({'k', 'n'}, {c.N, c.N}, c.dt), rng),
                D(colMajor({'m', 'n'}, {c.N, c.N}, c.dt), rng);
            double flops = (isCplx(c.dt) ? 8.0 : 2.0) * (double)c.N * c.N * c.N;
            Timing t = timeLaunch([&]() { cublasGemm(cb128, c.dt, c.N, A.buf->d, B.buf->d, D.buf->d); }, warm, iters, stream);
            Timing t2 = timeLaunch([&]() { cublasGemm(cb, c.dt, c.N, A.buf->d, B.buf->d, D.buf->d); }, warm, iters, stream);
            printf("[21] %s | cuBLAS 128MiB-ws median %.3f ms (%.3f TFLOP/s) | cuBLAS default-ws median %.3f ms (%.3f TFLOP/s)\n", c.label,
                   t.median, flops / t.median / 1e9, t2.median, flops / t2.median / 1e9);
        }
        CUDA_CHECK(cudaFree(ws2));
        cublasDestroy(cb128);
    }
    cublasDestroy(cb);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Check 22: D[m,n,p] = sum_{k1,k2} A[m,k1,k2] * B[k2,n,k1,p]  (p is a batch mode living only in B)
// variants: B rank-3 (p only in D -> invalid), p in A, packed vs arbitrary strides
enum class PWhere { POnlyInD, PInB, PInA };

static void nonGemmCase(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng, int sz, int k1,
                        int k2, PWhere pw, bool strided, bool verify, const char* id, int warm, int iters) {
    const int m = sz, n = sz, p = sz;
    TSpec sA, sB, sD;
    if (strided) {
        // A[m,k1,k2] at m*(k1*k2+3) + k1*(k2+1) + k2  (m slabs of a (k1, k2+1) col-major block plus padding)
        // D[m,n,p]   at m + n*(m+2) + p*(m+2)*(n+1)  (slice of a col-major (m+2, n+1, p) buffer)
        if (pw == PWhere::PInA) {
            sA = TSpec({'m', 'k', 'l', 'p'}, {m, k1, k2, p}, {k1 * k2 + 3, k2 + 1, 1, m * (k1 * k2 + 3)}, CUDA_R_64F, 8);
            sB = TSpec({'l', 'n', 'k'}, {k2, n, k1}, {1, k2 + 2, (k2 + 2) * (n + 1)}, CUDA_R_64F, 8);
        } else {
            sA = TSpec({'m', 'k', 'l'}, {m, k1, k2}, {k1 * k2 + 3, k2 + 1, 1}, CUDA_R_64F, 8);
            // B[k2,n,k1,p] at k2 + n*(k2+2) + k1*(k2+2)*(n+1) + p*(k2+2)*(n+1)*(k1+1)  (slice of col-major (k2+2, n+1, k1+1, p))
            if (pw == PWhere::PInB)
                sB = TSpec({'l', 'n', 'k', 'p'}, {k2, n, k1, p}, {1, k2 + 2, (k2 + 2) * (n + 1), (k2 + 2) * (n + 1) * (k1 + 1)},
                           CUDA_R_64F, 8);
            else
                sB = TSpec({'l', 'n', 'k'}, {k2, n, k1}, {1, k2 + 2, (k2 + 2) * (n + 1)}, CUDA_R_64F, 8);
        }
        sD = TSpec({'m', 'n', 'p'}, {m, n, p}, {1, m + 2, (m + 2) * (n + 1)}, CUDA_R_64F, 8);
    } else {
        if (pw == PWhere::PInA) {
            sA = colMajor({'m', 'k', 'l', 'p'}, {m, k1, k2, p});
            sB = colMajor({'l', 'n', 'k'}, {k2, n, k1});
        } else if (pw == PWhere::PInB) {
            sA = colMajor({'m', 'k', 'l'}, {m, k1, k2});
            sB = colMajor({'l', 'n', 'k', 'p'}, {k2, n, k1, p});
        } else {
            sA = colMajor({'m', 'k', 'l'}, {m, k1, k2});
            sB = colMajor({'l', 'n', 'k'}, {k2, n, k1});
        }
        sD = colMajor({'m', 'n', 'p'}, {m, n, p});
    }
    Tsr A(sA, rng), B(sB, rng), D(sD, rng), C(sD, rng);
    Scalars sc = scalarsF64(1.0, 0.0);
    // host planning cost for this problem: first time vs repeated (plan cache), pipeline without execution
    auto planOnly = [&]() {
        Stages s2;
        PlanKeep k2;
        auto t0 = clk::now();
        runContraction(h, A, ID, B, ID, D, ID, D, CUTENSOR_COMPUTE_DESC_64F, sc.alpha, sc.beta, ws, wsSize, stream, s2, &k2, -1, false);
        return msSince(t0, clk::now());
    };
    double planCold = planOnly();
    std::vector<double> planWarm(10);
    for (auto& w : planWarm) w = planOnly();
    std::sort(planWarm.begin(), planWarm.end());
    PlanKeep keep;
    Stages st;
    runContraction(h, A, ID, B, ID, D, ID, D, CUTENSOR_COMPUTE_DESC_64F, sc.alpha, sc.beta, ws, wsSize, stream, st, &keep, -1, true);
    double flops = 2.0 * m * n * p * k1 * k2;
    const char* where = pw == PWhere::PInB ? "B[k2,n,k1,p]" : pw == PWhere::PInA ? "A[m,k1,k2,p]" : "p only in D (invalid)";
    if (!st.ok()) {
        printf("[%s] ERR | stage=%s status=%d (%s) | %s %s, m=%d k1=%d k2=%d n=%d p=%d\n", id, st.firstFail().c_str(),
               (int)st.firstFailStatus(), sname(st.firstFailStatus()), strided ? "strided" : "packed", where, m, k1, k2, n, p);
        fflush(stdout);
        return;
    }
    std::string verdict = "not verified", detail = "";
    if (verify) {
        D.download();
        auto ext = modeExtents({&A.s, &B.s, &C.s, &D.s});
        std::vector<uint8_t> href = D.buf->h0;
        refContract<double>(A.buf->h0.data(), A.s, false, B.buf->h0.data(), B.s, false, 1.0, 0.0, C.buf->h0.data(), C.s, false,
                            href.data(), D.s, ext);
        double maxAbs = 0, maxRef = 0;
        for (int64_t o : touchedOffsets(D.s, ext)) {
            W g = readElem(D.buf->h.data(), D.s.dt, o), r = readElem(href.data(), D.s.dt, o);
            maxAbs = std::max(maxAbs, std::abs(g - r));
            maxRef = std::max(maxRef, std::abs(r));
        }
        double norm = maxAbs / maxRef;
        verdict = norm <= tolOf(CUDA_R_64F) ? "ok" : "FAIL";
        char buf[128];
        snprintf(buf, sizeof buf, "maxabs=%.3e maxabs/max|ref|=%.3e | ", maxAbs, norm);
        detail = buf;
    }
    Timing t = timeLaunch([&]() {
        cutensorContract(h, keep.plan, sc.alpha, A.buf->dptr(), B.buf->dptr(), sc.beta, D.buf->dptr(), D.buf->dptr(), ws, wsSize, stream);
    }, warm, iters, stream);
    printf("[%s] %s | %s%s, %s, m=%d k1=%d k2=%d n=%d p=%d | %.3f GFLOP | wsEstimate=%llu B wsRequired=%llu B | host plan: first %.3f ms, repeated median %.3f ms | device median %.4f ms (min %.4f max %.4f) = %.3f TFLOP/s\n",
           id, verdict.c_str(), detail.c_str(), strided ? "strided" : "packed col-major", where, m, k1, k2, n, p, flops / 1e9,
           (unsigned long long)st.wsEstimate, (unsigned long long)st.wsRequired, planCold, planWarm[5], t.median, t.min, t.max,
           flops / t.median / 1e9);
    fflush(stdout);
}

static void checkNonGemm(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng) {
    printf("--- check 22: non-GEMM contraction D[m,n,p] = sum_{k1,k2} A[m,k1,k2] * B[k2,n,k1(,p)] (f64) ---\n");
    nonGemmCase(h, stream, ws, wsSize, rng, 64, 32, 64, PWhere::POnlyInD, false, false, "22.0", 0, 1);
    nonGemmCase(h, stream, ws, wsSize, rng, 64, 32, 64, PWhere::PInB, true, true, "22a", 5, 50);
    nonGemmCase(h, stream, ws, wsSize, rng, 64, 32, 64, PWhere::PInB, false, true, "22b", 5, 50);
    nonGemmCase(h, stream, ws, wsSize, rng, 64, 32, 64, PWhere::PInA, true, true, "22c", 5, 50);
    nonGemmCase(h, stream, ws, wsSize, rng, 128, 32, 64, PWhere::PInB, true, false, "22d", 3, 10);
    nonGemmCase(h, stream, ws, wsSize, rng, 128, 32, 64, PWhere::PInB, false, false, "22e", 3, 10);
    nonGemmCase(h, stream, ws, wsSize, rng, 256, 32, 64, PWhere::PInB, true, false, "22f", 3, 5);
    nonGemmCase(h, stream, ws, wsSize, rng, 256, 32, 64, PWhere::PInB, false, false, "22g", 3, 5);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Check 23: many-mode contractions (rank 8 and 12 operands)
// Builds a strided spec: a slice of a column-major buffer whose extents are ext+pad.
static TSpec paddedSpec(std::vector<int32_t> modes, std::vector<int64_t> ext, std::vector<int64_t> pad) {
    std::vector<int64_t> bufExt(ext.size()), st(ext.size());
    int64_t acc = 1;
    for (size_t i = 0; i < ext.size(); ++i) {
        st[i] = acc;
        bufExt[i] = ext[i] + pad[i];
        acc *= bufExt[i];
    }
    return TSpec(modes, ext, st, CUDA_R_64F, 8);
}

static void manyModeCase(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng, const char* id,
                         TSpec sA, TSpec sB, TSpec sD, const char* note) {
    Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
    Scalars sc = scalarsF64(1.0, 0.5);
    CheckResult r = runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_64F, sc, true);
    report(id, r, CUDA_R_64F, note);
}

static void checkManyMode(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng) {
    printf("--- check 23: many-mode contractions (f64; extents 2..5) ---\n");
    {
        // rank-8 operands: A[a,b,c,d,p,q,r,s] x B[p,q,r,s,e,f,g,h] -> D[a,b,c,d,e,f,g,h] (rank-8 output, 4 contracted modes)
        std::vector<int32_t> mA = {1, 2, 3, 4, 5, 6, 7, 8}, mB = {5, 6, 7, 8, 9, 10, 11, 12}, mD = {1, 2, 3, 4, 9, 10, 11, 12};
        std::vector<int64_t> e = {2, 3, 4, 5, 2, 3, 4, 5};
        manyModeCase(h, stream, ws, wsSize, rng, "23a", colMajor(mA, e), colMajor(mB, e), colMajor(mD, e),
                     "rank-8 A,B,D: 4 contracted + 4 batch modes each side, packed; D = A*B + 0.5*C");
    }
    {
        std::vector<int32_t> mA = {1, 2, 3, 4, 5, 6, 7, 8}, mB = {5, 6, 7, 8, 9, 10, 11, 12}, mD = {1, 2, 3, 4, 9, 10, 11, 12};
        std::vector<int64_t> e = {2, 3, 4, 5, 2, 3, 4, 5};
        std::vector<int64_t> pad = {1, 0, 2, 1, 0, 1, 0, 3};
        manyModeCase(h, stream, ws, wsSize, rng, "23b", paddedSpec(mA, e, pad), paddedSpec(mB, e, pad), paddedSpec(mD, e, pad),
                     "rank-8, every tensor a padded slice (strides from extents+pad, e.g. 3,4,8,12,...), align 8");
    }
    {
        // shared batch modes: A[a,b,c,d,p,q,r,s] x B[c,d,e,f,p,q,r,s] -> D[a,b,c,d,e,f]
        std::vector<int32_t> mA = {1, 2, 3, 4, 5, 6, 7, 8}, mB = {3, 4, 9, 10, 5, 6, 7, 8}, mD = {1, 2, 3, 4, 9, 10};
        std::vector<int64_t> eA = {2, 3, 4, 5, 2, 3, 4, 5}, eB = {4, 5, 2, 3, 2, 3, 4, 5}, eD = {2, 3, 4, 5, 2, 3};
        manyModeCase(h, stream, ws, wsSize, rng, "23c", colMajor(mA, eA), colMajor(mB, eB), colMajor(mD, eD),
                     "rank-8 operands with shared batch modes c,d; 4 contracted; rank-6 output");
    }
    {
        // rank-12 operands, extents 2 and 3
        std::vector<int32_t> mA, mB, mD;
        std::vector<int64_t> eA, eB, eD;
        for (int i = 1; i <= 6; ++i) { mA.push_back(i); eA.push_back(i % 2 ? 2 : 3); mD.push_back(i); eD.push_back(i % 2 ? 2 : 3); }
        for (int i = 7; i <= 12; ++i) { mA.push_back(i); eA.push_back(i % 2 ? 3 : 2); mB.push_back(i); eB.push_back(i % 2 ? 3 : 2); }
        for (int i = 13; i <= 18; ++i) { mB.push_back(i); eB.push_back(i % 2 ? 2 : 3); mD.push_back(i); eD.push_back(i % 2 ? 2 : 3); }
        std::vector<int64_t> pad(12, 1);
        manyModeCase(h, stream, ws, wsSize, rng, "23d", paddedSpec(mA, eA, pad), paddedSpec(mB, eB, pad), paddedSpec(mD, eD, pad),
                     "rank-12 operands (6 contracted modes), rank-12 output, extents 2/3, padded-slice strides");
    }
    {
        // rank-8 with permuted mode order in every operand (non-canonical)
        std::vector<int32_t> mA = {8, 6, 4, 2, 7, 5, 3, 1}, mB = {11, 5, 9, 7, 12, 6, 10, 8}, mD = {12, 10, 8, 6, 4, 2, 1, 3};
        std::vector<int64_t> e = {5, 3, 5, 3, 4, 2, 4, 2};  // extents follow the permuted order (mode i has extent 2+(i%4))
        auto extOf = [](int32_t m) { return (int64_t)(2 + (m - 1) % 4); };
        for (size_t i = 0; i < mA.size(); ++i) e[i] = extOf(mA[i]);
        std::vector<int64_t> eB(mB.size());
        for (size_t i = 0; i < mB.size(); ++i) eB[i] = extOf(mB[i]);
        std::vector<int64_t> eD(mD.size());
        for (size_t i = 0; i < mD.size(); ++i) eD[i] = extOf(mD[i]);
        std::vector<int64_t> pad(8, 1);
        manyModeCase(h, stream, ws, wsSize, rng, "23e", paddedSpec(mA, e, pad), paddedSpec(mB, eB, pad), paddedSpec(mD, eD, pad),
                     "rank-8, modes permuted differently in A, B and D; extents 2..5; padded-slice strides");
    }
    fflush(stdout);
}

void planSequence(cutensorHandle_t h, int n, char** sizes) {
    const uint64_t WS = 128ull << 20;
    for (int i = 0; i < n; ++i) {
        int N = atoi(sizes[i]);
        PlanTiming t = planOnce(h, N, N, N, CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, -1, WS);
        char lbl[128];
        snprintf(lbl, sizeof lbl, "seq#%d size %d", i + 1, N);
        printPlanTiming(lbl, t);
    }
}

void checkPerf(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, bool includeCorrectness) {
    std::mt19937_64 rng(20261127);
    checkPlanning(h);
    checkGemmPerf(h, stream, ws, wsSize, rng);
    checkNonGemm(h, stream, ws, wsSize, rng);
    checkManyMode(h, stream, ws, wsSize, rng);
    (void)includeCorrectness;
}
