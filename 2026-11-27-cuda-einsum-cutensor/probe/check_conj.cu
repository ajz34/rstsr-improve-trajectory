// Checks 10-17: conjugate operands, opC, scalar types, mixed real/complex.
#include "probe_common.h"

// Max abs error of the downloaded D against a host reference buffer.
static double maxErrVs(const Tsr& D, const std::vector<uint8_t>& href, const std::map<int32_t, int64_t>& ext, double& maxRef) {
    double e = 0;
    maxRef = 0;
    for (int64_t o : touchedOffsets(D.s, ext)) {
        W got = readElem(D.buf->h.data(), D.s.dt, o + D.buf->baseOff);
        W ref = readElem(href.data(), D.s.dt, o + D.buf->baseOff);
        e = std::max(e, std::abs(got - ref));
        maxRef = std::max(maxRef, std::abs(ref));
    }
    return e;
}

// Check 14: opC = CONJ with beta != 0; compare against both conj(C) and plain-C references.
template <class ACC>
static void checkOpCConj(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, std::mt19937_64& rng,
                         cudaDataType_t dt, cutensorComputeDescriptor_t comp, const Scalars& sc, const char* id) {
    Tsr A(colMajor({'m', 'k'}, {M0, K0}, dt), rng), B(colMajor({'k', 'n'}, {K0, N0}, dt), rng),
        C(colMajor({'m', 'n'}, {M0, N0}, dt), rng), D(colMajor({'m', 'n'}, {M0, N0}, dt), rng);
    Stages st;
    runContraction(h, A, ID, B, ID, C, CJ, D, comp, sc.alpha, sc.beta, ws, wsSize, stream, st);
    CheckResult r;
    r.st = st;
    if (!st.ok()) {
        r.anyErr = true;
        r.errStage = st.firstFail();
        r.errCode = (int)st.firstFailStatus();
        char buf[128];
        snprintf(buf, sizeof buf, "status=%d (%s)", r.errCode, sname(st.firstFailStatus()));
        r.errStr = buf;
        report(id, r, dt, "opC=CONJ, beta=0.75 (cplx): D = A*B + 0.75*conj(C) requested; status of op creation", true);
        return;
    }
    D.download();
    auto ext = modeExtents({&A.s, &B.s, &C.s, &D.s});
    std::vector<uint8_t> h1 = D.buf->h0, h2 = D.buf->h0;
    refContract<ACC>(A.buf->h0.data(), A.s, false, B.buf->h0.data(), B.s, false, conv<ACC>(sc.alphaV), conv<ACC>(sc.betaV),
                     C.buf->h0.data(), C.s, true, h1.data(), D.s, ext);
    refContract<ACC>(A.buf->h0.data(), A.s, false, B.buf->h0.data(), B.s, false, conv<ACC>(sc.alphaV), conv<ACC>(sc.betaV),
                     C.buf->h0.data(), C.s, false, h2.data(), D.s, ext);
    double mr1 = 0, mr2 = 0;
    double e1 = maxErrVs(D, h1, ext, mr1), e2 = maxErrVs(D, h2, ext, mr2);
    double tol = tolOf(dt) * std::max(mr1, 1e-300);
    std::string v, detail;
    char buf[256];
    if (e1 <= tol) {
        v = "ok";
        snprintf(buf, sizeof buf, "maxabs_vs_conjC=%.3e maxabs_vs_plainC=%.3e", e1, e2);
        detail = std::string(buf) + " | opC=CONJ honored: D = A*B + 0.75*conj(C)";
    } else if (e2 <= tol) {
        v = "FAIL (opC silently ignored)";
        snprintf(buf, sizeof buf, "maxabs_vs_conjC=%.3e maxabs_vs_plainC=%.3e", e1, e2);
        detail = std::string(buf) + " | accepted but computed D = A*B + 0.75*C (conj NOT applied)";
    } else {
        v = "FAIL (matches neither)";
        snprintf(buf, sizeof buf, "maxabs_vs_conjC=%.3e maxabs_vs_plainC=%.3e", e1, e2);
        detail = buf;
    }
    r.haveResult = true;
    printf("[%s] %s | %s | %s\n", id, v.c_str(), detail.c_str(), "opC=CONJ, beta=0.75: status SUCCESS at op creation; result checked vs conj(C) and vs C");
    fflush(stdout);
}

void checkConj(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize) {
    std::mt19937_64 rng(20261127);
    struct DT { cudaDataType_t dt; cutensorComputeDescriptor_t comp; const char* name; };
    const DT dts[2] = {{CUDA_C_64F, CUTENSOR_COMPUTE_DESC_64F, "c64"}, {CUDA_C_32F, CUTENSOR_COMPUTE_DESC_32F, "c32"}};
    const int m = M0, k = K0, n = N0;

    for (int t = 0; t < 2; ++t) {
        cudaDataType_t dt = dts[t].dt;
        cutensorComputeDescriptor_t comp = dts[t].comp;
        const char* nm = dts[t].name;
        std::string tag = std::string(t == 0 ? "" : " (") + (t == 0 ? "" : nm) + (t == 0 ? "" : ")");
        Scalars sc0 = (t == 0) ? scalarsC64(W(1, 0), W(0, 0)) : scalarsC32(W(1, 0), W(0, 0));
        Scalars sc5 = (t == 0) ? scalarsC64(W(1, 0), W(0.5, 0)) : scalarsC32(W(1, 0), W(0.5, 0));
        Scalars sc75 = (t == 0) ? scalarsC64(W(1, 0), W(0.75, 0)) : scalarsC32(W(1, 0), W(0.75, 0));
        Scalars sca = (t == 0) ? scalarsC64(W(1.5, -0.5), W(0.5, 0.25)) : scalarsC32(W(1.5, -0.5), W(0.5, 0.25));
        std::string note;
        printf("--- checks 10-12, 14-17 with %s (alpha/beta = %s) ---\n", nm, sc0.desc.c_str());

        {
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            report("10" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, ID, C, ID, D, comp, sc0, false), dt,
                   "opA=CONJ, opB=ID: D = conj(A) @ B, packed; complex random data");
        }
        {
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            report("11" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, CJ, C, ID, D, comp, sc0, false), dt,
                   "opA=ID, opB=CONJ: D = A @ conj(B), packed");
        }
        {
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            report("12" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, CJ, C, ID, D, comp, sca, false), dt,
                   "opA=CONJ, opB=CONJ: D = (1.5-0.5i)*conj(A) @ conj(B) + (0.5+0.25i)*C (complex alpha/beta)");
        }
        {
            // permuted modes + conj
            Tsr A(colMajor({'k', 'm'}, {k, m}, dt), rng), B(colMajor({'n', 'k'}, {n, k}, dt), rng),
                C(colMajor({'n', 'm'}, {n, m}, dt), rng), D(colMajor({'n', 'm'}, {n, m}, dt), rng, false);
            report("12b" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, CJ, C, ID, D, comp, sc0, false), dt,
                   "opA=CONJ, opB=CONJ with permuted modes modeA={k,m} modeB={n,k} modeD={n,m}");
        }

        if (t == 0) {
            printf("--- check 13: opA = CONJ on a real f64 tensor ---\n");
            Scalars sr = scalarsF64(1.0, 0.0);
            Tsr A(colMajor({'m', 'k'}, {m, k}, CUDA_R_64F), rng), B(colMajor({'k', 'n'}, {k, n}, CUDA_R_64F), rng),
                C(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng), D(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng, false);
            report("13a", runTyped(h, stream, ws, wsSize, A, CJ, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_64F, sr, false),
                   CUDA_R_64F, "opA=CONJ on CUDA_R_64F (docs table: CONJ is for complex types); record status", true);
            Tsr A2(colMajor({'m', 'k'}, {m, k}, CUDA_R_64F), rng), B2(colMajor({'k', 'n'}, {k, n}, CUDA_R_64F), rng),
                C2(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng), D2(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng, false);
            report("13b", runTyped(h, stream, ws, wsSize, A2, ID, B2, CJ, C2, ID, D2, CUTENSOR_COMPUTE_DESC_64F, sr, false),
                   CUDA_R_64F, "opB=CONJ on CUDA_R_64F; record status", true);
        }

        {
            // check 14
            if (t == 0) checkOpCConj<W>(h, stream, ws, wsSize, rng, dt, comp, sc75, "14a");
            else checkOpCConj<CF>(h, stream, ws, wsSize, rng, dt, comp, sc75, "14a (c32)");
            // also with beta real-zero imaginary part but C complex: same thing. And opC=CONJ with beta=0 (C unused)
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            report("14b" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, CJ, D, comp, sc0, false), dt,
                   "opC=CONJ with beta=0 (C not read): record status; result must equal A@B", true);
        }

        {
            // check 15: conj on strided operands (layout of check 1.1)
            TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 2}, dt, 16);
            TSpec sB({'k', 'n'}, {k, n}, {1, k + 5}, dt, 16);
            TSpec sD({'m', 'n'}, {m, n}, {2 * n, 1}, dt, 16);
            if (t == 0) {
                // the first run (align 8 declared for a 16-byte element type) was rejected at descA: record explicitly
                TSpec sA8({'m', 'k'}, {m, k}, {2 * k + 3, 2}, dt, 8);
                TSpec sB8({'k', 'n'}, {k, n}, {1, k + 5}, dt, 8);
                TSpec sD8({'m', 'n'}, {m, n}, {2 * n, 1}, dt, 8);
                Tsr A(sA8, rng), B(sB8, rng), D(sD8, rng, false), C(sD8, rng);
                report("15a-align8" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, ID, C, ID, D, comp, sc0, false), dt,
                       "same strided layout but alignmentRequirement = 8 declared for CUDA_C_64F (16-byte elements)", true);
            }
            {
                Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
                report("15a" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, ID, C, ID, D, comp, sc0, false), dt,
                       "conj(A_strided) @ B_strided; A (109,2) B (1,58) D (82,1); beta=0");
            }
            {
                Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
                report("15b" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, CJ, C, ID, D, comp, sc0, false), dt,
                       "A_strided @ conj(B_strided); beta=0");
            }
            {
                Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
                report("15c" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, CJ, C, ID, D, comp, sc5, false), dt,
                       "conj(A_strided) @ conj(B_strided) + 0.5*C_strided (C separate, same layout as D)");
            }
            {
                // transposed (permuted-mode) complex operands with conj
                TSpec sA2({'k', 'm'}, {k, m}, {2, 2 * k}, dt, 16);
                TSpec sB2({'n', 'k'}, {n, k}, {2, 4 * k}, dt, 16);
                TSpec sD2({'n', 'm'}, {n, m}, {2, 4 * m}, dt, 16);
                Tsr A(sA2, rng), B(sB2, rng), D(sD2, rng, false), C(sD2, rng);
                report("15d" + tag, runTyped(h, stream, ws, wsSize, A, CJ, B, CJ, C, ID, D, comp, sc5, false), dt,
                       "conj(A{k,m}) @ conj(B{n,k}) -> D{n,m}, all modes non-contiguous (2x interleaved); beta=0.5");
            }
        }

        if (t == 0) {
            printf("--- check 16: scalar pointer type for c64 (COMPUTE_DESC_64F) ---\n");
            {
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("16a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, false), dt,
                       "alpha/beta = cuDoubleComplex{1,0}/{0,0} (documented type) -- reference");
            }
            {
                double* a = new double[2]{1.0, 0.0};
                double* b = new double[2]{0.0, 0.0};
                Scalars sc;
                sc.alpha = a;
                sc.beta = b;
                sc.alphaV = W(1, 0);
                sc.betaV = W(0, 0);
                sc.desc = "double[2]{1,0}";
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("16b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc, false), dt,
                       "alpha/beta passed as double[2]{1,0}/{0,0} (same bytes as cuDoubleComplex): no error possible");
            }
            {
                double* a = new double[2]{1.0, 0.25};
                double* b = new double[2]{0.0, 0.0};
                Scalars sc;
                sc.alpha = a;
                sc.beta = b;
                sc.alphaV = W(1, 0.25);
                sc.betaV = W(0, 0);
                sc.desc = "double[2]{1,0.25}";
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("16c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc, false), dt,
                       "alpha passed as double[2]{1,0.25}: reference assumes it is read as (1+0.25i); 'ok' = silently interpreted as complex");
            }
            {
                float* a = new float[4]{1.0f, 0.0f, 0.0f, 0.0f};
                float* b = new float[4]{0.0f, 0.0f, 0.0f, 0.0f};
                Scalars sc;
                sc.alpha = a;
                sc.beta = b;
                sc.alphaV = W(1, 0);
                sc.betaV = W(0, 0);
                sc.desc = "float[4]{1,0,0,0}";
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("16d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc, false), dt,
                       "alpha passed as float[4]{1,0,0,0} (WRONG type; 16 bytes read as two doubles): reference assumes alpha=1");
            }
            {
                // scalar-type attribute: query default, then set CUDA_R_64F and pass a plain double
                cudaDataType_t queried = (cudaDataType_t)-1, setTo = CUDA_R_64F;
                cutensorStatus_t gs = CUTENSOR_STATUS_NOT_INITIALIZED, ss = CUTENSOR_STATUS_NOT_INITIALIZED;
                OpHook hook = [&](cutensorOperationDescriptor_t op, Stages& st) {
                    gs = cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &queried, sizeof(queried));
                    ss = cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &setTo, sizeof(setTo));
                    printf("   16e: SCALAR_TYPE default = %d (%s) [Get status %d], Set to CUDA_R_64F -> status %d (%s)\n",
                           (int)queried, queried == CUDA_C_64F ? "CUDA_C_64F" : queried == CUDA_R_64F ? "CUDA_R_64F" : "other",
                           (int)gs, (int)ss, sname(ss));
                    if (!isok(ss)) {
                        st.op = ss;  // stop if the set failed
                        st.hookFailStage = "SetAttribute(SCALAR_TYPE)";
                    }
                };
                Scalars sc = scalarsF64(1.0, 0.0);
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("16e", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc, false, hook), dt,
                       "after SetAttribute(SCALAR_TYPE=CUDA_R_64F), alpha/beta passed as plain double (reference alpha=1 real)", true);
            }
        }

        {
            printf("--- check 17: mixed real/complex (%s) ---\n", nm);
            cudaDataType_t rt = (t == 0) ? CUDA_R_64F : CUDA_R_32F;
            Scalars scr = (t == 0) ? scalarsC64(W(1, 0), W(0.5, 0)) : scalarsC32(W(1, 0), W(0.5, 0));
            {
                Tsr A(colMajor({'m', 'k'}, {m, k}, rt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                cudaDataType_t queried = (cudaDataType_t)-1;
                OpHook hook = [&](cutensorOperationDescriptor_t op, Stages&) {
                    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &queried, sizeof(queried));
                    printf("   17a%s: SCALAR_TYPE attribute of the mixed problem = %d (%s)\n", tag.c_str(), (int)queried,
                           queried == CUDA_C_64F ? "CUDA_C_64F" : queried == CUDA_C_32F ? "CUDA_C_32F" : queried == CUDA_R_64F ? "CUDA_R_64F"
                                                                                                                   : "other");
                };
                report("17a" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, scr, false, hook), dt,
                       "A real, B complex -> C/D complex; beta=0.5 (complex beta) with C complex", t == 1);
            }
            {
                Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, rt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("17b" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, scr, false), dt,
                       "A complex, B real -> C/D complex; beta=0.5", t == 1);
            }
            {
                // real operand + CONJ on the complex operand
                Tsr A(colMajor({'m', 'k'}, {m, k}, rt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("17c" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, CJ, C, ID, D, comp, scr, false), dt,
                       "A real, opB=CONJ on complex B -> D = A @ conj(B) + 0.5*C", t == 1);
            }
            {
                // mixed + strided (release notes 1.x: fixed bug with mixed real-complex contraction and strided data)
                TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 2}, rt, 8);
                TSpec sB({'k', 'n'}, {k, n}, {1, k + 5}, dt, 16);
                TSpec sD({'m', 'n'}, {m, n}, {2 * n, 1}, dt, 16);
                Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
                report("17d" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, scr, false), dt,
                       "A real strided (109,2), B complex strided (1,58), D complex strided (82,1); beta=0.5", t == 1);
            }
            {
                // real output (not in the docs table)
                Tsr A(colMajor({'m', 'k'}, {m, k}, rt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, rt), rng), D(colMajor({'m', 'n'}, {m, n}, rt), rng, false);
                report("17e" + tag, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, scr, false), rt,
                       "A real, B complex -> C/D REAL (not in docs table); reference = complex result with imaginary part dropped",
                       true);
            }
            if (t == 1) {
                // real scalar pointers with mixed complex 32-bit (docs: typeScalar = CUDA_C_32F for C32xC32)
                float* fa = new float[2]{1.0f, 0.5f};
                float* fb = new float[2]{0.5f, 0.0f};
                Scalars scf;
                scf.alpha = fa;
                scf.beta = fb;
                scf.alphaV = W(1, 0);
                scf.betaV = W(0.5, 0);
                scf.desc = "float[2]{1,0.5}/float[2]{0.5,0}";
                Tsr A(colMajor({'m', 'k'}, {m, k}, rt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                    C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
                report("17f (c32)", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, scf, false), dt,
                       "R32 x C32 -> C32 with alpha/beta passed as float[2]{1,0.5}/{0.5,0} (wrong type: read as alpha=1+0.5i, beta=0.5); reference alpha=1", true);
            }
        }
    }
}
