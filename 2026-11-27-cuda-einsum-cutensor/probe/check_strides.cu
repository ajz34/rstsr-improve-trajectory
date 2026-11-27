// Checks 1-9: strides, broadcasting, extent-1, negative stride, extent 0, alignment, rank-0, duplicate modes.
#include "probe_common.h"

void checkStrides(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize) {
    std::mt19937_64 rng(20261127);
    const int m = M0, k = K0, n = N0;
    const cutensorComputeDescriptor_t comp = CUTENSOR_COMPUTE_DESC_64F;
    auto sc0 = scalarsF64(1.0, 0.0);
    auto sc5 = scalarsF64(1.0, 0.5);
    const W SENT(-7.5e300, 0.0);

    printf("--- check 1: arbitrary positive strides (f64, C[m,n]=sum_k A[m,k]B[k,n], m=%d k=%d n=%d) ---\n", m, k, n);
    {
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("1.0", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "baseline: packed col-major A,B,D; beta=0; align=256");
    }
    {
        // A = every 2nd column of a row-major (m, 2k+3) buffer; B = transposed view of col-major (k+5, n);
        // D = every 2nd row of a row-major (2m, n) buffer; C separate with D's layout
        TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 2}, CUDA_R_64F, 8);
        TSpec sB({'k', 'n'}, {k, n}, {1, k + 5}, CUDA_R_64F, 16);
        TSpec sD({'m', 'n'}, {m, n}, {2 * n, 1}, CUDA_R_64F, 16);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("1.1", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "A strides(109,2) align8; B strides(1,58) align16; D strides(82,1) align16; beta=0.5, C separate");
    }
    {
        // all strides odd / not 16-byte multiples
        TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 3}, CUDA_R_64F, 8);
        TSpec sB({'k', 'n'}, {k, n}, {1, k + 4}, CUDA_R_64F, 8);
        TSpec sD({'m', 'n'}, {m, n}, {2 * n + 1, 2}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("1.2", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A strides(109,3) B strides(1,57) D strides(83,2); none a 16B multiple; align=8; beta=0");
    }
    {
        // in-place accumulation: C aliases D (beta = 0.5)
        TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 2}, CUDA_R_64F, 8);
        TSpec sB({'k', 'n'}, {k, n}, {1, k + 5}, CUDA_R_64F, 16);
        TSpec sD({'m', 'n'}, {m, n}, {2 * n, 1}, CUDA_R_64F, 16);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng), C(sD, D);
        report("1.3", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "same layout as 1.1 but C aliases D (in-place D = A*B + 0.5*D)");
    }
    {
        // permuted modes, all packed
        Tsr A(colMajor({'k', 'm'}, {k, m}), rng), B(colMajor({'n', 'k'}, {n, k}), rng), C(colMajor({'n', 'm'}, {n, m}), rng),
            D(colMajor({'n', 'm'}, {n, m}), rng, false, SENT);
        report("1.4", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "modeA={k,m} modeB={n,k} modeD={n,m}, all packed col-major (D^T = B^T A^T)");
    }
    {
        // every mode of every tensor non-contiguous (2x interleaved views)
        TSpec sA({'m', 'k'}, {m, k}, {2, 2 * m}, CUDA_R_64F, 16);
        TSpec sB({'k', 'n'}, {k, n}, {2, 4 * k}, CUDA_R_64F, 16);
        TSpec sD({'m', 'n'}, {m, n}, {2, 4 * m}, CUDA_R_64F, 16);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("1.5", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "A strides(2,74) B strides(2,212) D strides(2,148): no mode contiguous; beta=0.5");
    }

    printf("--- check 2: stride zero (extent > 1) ---\n");
    {
        TSpec sA({'m', 'k'}, {m, k}, {1, 0});
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("2a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A[m,k] with stride_k = 0 (A[m,k]=a[m]); expect INVALID_VALUE at descA", true);
    }
    {
        TSpec sB({'k', 'n'}, {k, n}, {1, 0});
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(sB, rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("2b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "B[k,n] with stride_n = 0 (B[k,n]=b[k]); expect INVALID_VALUE at descB", true);
    }
    {
        TSpec sD({'m', 'n'}, {m, n}, {1, 0});
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("2c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "D[m,n] with stride_n = 0 (output broadcast); expect INVALID_VALUE at descD", true);
    }
    {
        TSpec sA({'m', 'k'}, {m, k}, {0, 1});
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("2d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A[m,k] with stride_m = 0 on the non-contracted (batch) mode; expect INVALID_VALUE at descA", true);
    }

    {
        // stride-0 broadcast at GEMM size (cuTENSOR routes such problems to cuBLAS): does acceptance persist?
        const int M = 256;
        Scalars s1 = scalarsF64(1.0, 0.0);
        Tsr A(TSpec({'m', 'k'}, {M, M}, {1, 0}), rng), B(colMajor({'k', 'n'}, {M, M}), rng), C(colMajor({'m', 'n'}, {M, M}), rng),
            D(colMajor({'m', 'n'}, {M, M}), rng, false);
        report("2e", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_64F, s1, false), CUDA_R_64F,
               "256^3 f64: A[m,k] with stride_k = 0 (A[m,k] = a[m]); status + correctness vs broadcast reference", true);
        Tsr A2(TSpec({'m', 'k'}, {M, M}, {0, 1}), rng), B2(colMajor({'k', 'n'}, {M, M}), rng), C2(colMajor({'m', 'n'}, {M, M}), rng),
            D2(colMajor({'m', 'n'}, {M, M}), rng, false);
        report("2f", runTyped(h, stream, ws, wsSize, A2, ID, B2, ID, C2, ID, D2, CUTENSOR_COMPUTE_DESC_64F, s1, false), CUDA_R_64F,
               "256^3 f64: A[m,k] with stride_m = 0 on the batch mode (all rows equal); status + correctness", true);
        Tsr A3(TSpec({'b', 'm', 'k'}, {3, M, M}, {0, 1, M}), rng), B3(colMajor({'b', 'k', 'n'}, {3, M, M}), rng),
            C3(colMajor({'b', 'm', 'n'}, {3, M, M}), rng), D3(colMajor({'b', 'm', 'n'}, {3, M, M}), rng, false);
        report("2g", runTyped(h, stream, ws, wsSize, A3, ID, B3, ID, C3, ID, D3, CUTENSOR_COMPUTE_DESC_64F, s1, false), CUDA_R_64F,
               "batched 3x256^3 f64: A[b,m,k] with stride_b = 0 (A broadcast over the batch); status + correctness", true);
    }

    printf("--- check 3: broadcast by omitting a mode ---\n");
    {
        Tsr A(colMajor({'a'}, {37}), rng), B(colMajor({'b'}, {41}), rng), C(colMajor({'a', 'b'}, {37, 41}), rng),
            D(colMajor({'a', 'b'}, {37, 41}), rng, false, SENT);
        report("3a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "outer product C[a,b] = A[a]*B[b] (A rank-1, B rank-1, no contracted mode)");
    }
    {
        Tsr A(colMajor({'a'}, {37}), rng), B(colMajor({'a', 'b'}, {37, 41}), rng), C(colMajor({'a', 'b'}, {37, 41}), rng),
            D(colMajor({'a', 'b'}, {37, 41}), rng, false, SENT);
        report("3b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "C[a,b] = A[a]*B[a,b] (A rank-1 shares batch mode a); beta=0.5");
    }
    {
        Tsr A(colMajor({'a', 'k'}, {37, 53}), rng), B(colMajor({'k'}, {53}), rng), C(colMajor({'a'}, {37}), rng),
            D(colMajor({'a'}, {37}), rng, false, SENT);
        report("3c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "matvec D[a] = sum_k A[a,k]*B[k] (B has only a contracted mode); beta=0.5");
    }
    {
        Tsr A(colMajor({'a', 'b'}, {37, 41}), rng), B(colMajor({'b'}, {41}), rng), C(colMajor({'a'}, {37}), rng),
            D(colMajor({'a'}, {37}), rng, false, SENT);
        report("3d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "D[a] = sum_b A[a,b]*B[b] (A batch over a, B rank-1 contracted); beta=0.5");
    }
    {
        // strided broadcast: A rank-1 every 2nd element, B col-major, D every 2nd row
        TSpec sA({'a'}, {37}, {2}, CUDA_R_64F, 16);
        TSpec sB({'a', 'b'}, {37, 41}, {1, 37}, CUDA_R_64F, 256);
        TSpec sD({'a', 'b'}, {37, 41}, {82, 1}, CUDA_R_64F, 16);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("3e", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "C[a,b] = A[a](stride 2) * B[a,b] into strided D (82,1); beta=0.5");
    }

    printf("--- check 4: extent-1 mode ---\n");
    {
        TSpec sA({'m', 'k'}, {1, k}, {3, 1});
        TSpec sD({'m', 'n'}, {1, n}, {1, 1});
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("4a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A ext (1,53) strides (3,1) [non-zero stride on extent-1 mode]; D ext (1,41) strides (1,1)");
    }
    {
        TSpec sA({'m', 'k'}, {1, k}, {0, 1});
        TSpec sD({'m', 'n'}, {1, n}, {1, 1});
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("4b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "same with stride 0 on the extent-1 mode; docs say every stride must be > 0", true);
    }
    {
        TSpec sA({'b', 'k'}, {1, k}, {3, 1});
        TSpec sB({'b', 'k', 'n'}, {1, k, n}, {7, 1, k});
        TSpec sD({'b', 'n'}, {1, n}, {1, 1});
        Tsr A(sA, rng), B(sB, rng), C(sD, rng), D(sD, rng, false, SENT);
        report("4c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "batch-of-1 mode b in A, B and D (A (3,1), B (7,1,53), D (1,1))");
    }
    {
        // extent-1 mode present only in A (a "contracted" mode of extent 1)
        TSpec sA({'b', 'k'}, {1, k}, {3, 1});
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'n'}, {n}), rng),
            D(colMajor({'n'}, {n}), rng, false, SENT);
        report("4d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A ext (1,53) strides (3,1), mode b absent from B and D: D[n] = sum_{b,k} A[b,k]*B[k,n]");
    }

    printf("--- check 5: negative stride (reversed view) ---\n");
    {
        // A reversed along k: A[m,k] = buf[m + (k-1-k)*m], base at (k-1)*m
        TSpec sA({'m', 'k'}, {m, k}, {1, -m}, CUDA_R_64F, 256);
        sA.baseOff = (k - 1) * (int64_t)m;
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("5a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A strides (1,-37) with base offset 1924 (A[:, ::-1]); record status", true);
    }
    {
        TSpec sD({'m', 'n'}, {m, n}, {-1, m}, CUDA_R_64F, 256);
        sD.baseOff = m - 1;
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("5b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "D strides (-1,37) with base offset 36 (D[::-1, :] output view); record status", true);
    }
    {
        TSpec sA({'m', 'k'}, {m, k}, {-1, -m}, CUDA_R_64F, 256);
        sA.baseOff = (m - 1) + (k - 1) * (int64_t)m;
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        report("5c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A strides (-1,-37) (both modes reversed); record status", true);
    }

    printf("--- check 6: extent 0 ---\n");
    {
        TSpec sA({'m', 'k'}, {m, 0}, {1, m});
        Tsr A(sA, rng, false, SENT), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
            D(colMajor({'m', 'n'}, {m, n}), rng, false, SENT);
        auto r = runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, false);
        report("6a", r, CUDA_R_64F, "A extent (37,0) [contracted mode 0 in A, 53 in B]; expect INVALID_VALUE", true);
    }
    {
        TSpec sA({'m', 'k'}, {0, k}, {1, 0});
        TSpec sD({'m', 'n'}, {0, n}, {1, 0});
        Tsr A(sA, rng, false, SENT), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng, false, SENT), D(sD, rng, false, SENT);
        auto r = runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, false);
        report("6b", r, CUDA_R_64F, "A extent (0,53), D extent (0,41) [batch mode 0]; expect INVALID_VALUE", true);
    }
    {
        TSpec sD({'m', 'n'}, {m, 0}, {1, m});
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng, false, SENT),
            D(sD, rng, false, SENT);
        auto r = runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, false);
        report("6c", r, CUDA_R_64F, "D extent (37,0) [output mode 0, not contracted]; expect INVALID_VALUE", true);
    }

    printf("--- check 7: alignment declaration (f64, 8 bytes/element) ---\n");
    {
        TSpec sA(colMajor({'m', 'k'}, {m, k}, CUDA_R_64F, 8));
        sA.baseOff = 1;
        TSpec sD(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F, 8));
        sD.baseOff = 1;
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("7a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "A and D base = cudaMalloc base + 8 bytes; alignmentRequirement = 8");
    }
    {
        TSpec sA(colMajor({'m', 'k'}, {m, k}, CUDA_R_64F, 256));
        sA.baseOff = 1;
        TSpec sD(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F, 256));
        sD.baseOff = 1;
        Tsr A(sA, rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(sD, rng), D(sD, rng, false, SENT);
        report("7b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "same 8-byte-aligned pointers, alignmentRequirement declared 256 (over-declared)");
    }
    {
        TSpec sA({'m', 'k'}, {m, k}, {3, 3 * m}, CUDA_R_64F, 8);
        TSpec sB({'k', 'n'}, {k, n}, {3, 3 * k}, CUDA_R_64F, 8);
        TSpec sD({'m', 'n'}, {m, n}, {3 * n, 3}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("7c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "strides with a 3-element (24 B) factor: A (3,111) B (3,159) D (123,3); alignmentRequirement = 8; beta=0.5");
    }
    {
        TSpec sA({'m', 'k'}, {m, k}, {3, 3 * m}, CUDA_R_64F, 256);
        TSpec sB({'k', 'n'}, {k, n}, {3, 3 * k}, CUDA_R_64F, 256);
        TSpec sD({'m', 'n'}, {m, n}, {3 * n, 3}, CUDA_R_64F, 256);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("7d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "same strides as 7c (24 B multiples, not 256 B) but alignmentRequirement declared 256; beta=0.5");
    }

    {
        // alignment requirement smaller than the element size
        struct C { cudaDataType_t dt; uint32_t al; const char* lbl; };
        const C cs[6] = {{CUDA_R_64F, 4, "f64 align 4"}, {CUDA_R_64F, 8, "f64 align 8"},
                         {CUDA_C_64F, 8, "c64 align 8"}, {CUDA_C_64F, 16, "c64 align 16"},
                         {CUDA_R_32F, 2, "f32 align 2"}, {CUDA_R_32F, 4, "f32 align 4"}};
        printf("--- check 7e: alignmentRequirement below the element size ---\n");
        for (const C& c : cs) {
            Tsr A(colMajor({'m', 'k'}, {m, k}, c.dt, c.al), rng), B(colMajor({'k', 'n'}, {k, n}, c.dt, c.al), rng),
                C(colMajor({'m', 'n'}, {m, n}, c.dt, c.al), rng), D(colMajor({'m', 'n'}, {m, n}, c.dt, c.al), rng, false);
            Scalars sc = (c.dt == CUDA_R_64F) ? scalarsF64(1.0, 0.0) : (c.dt == CUDA_R_32F) ? scalarsF32(1.0f, 0.0f) : scalarsC64(W(1, 0), W(0, 0));
            cutensorComputeDescriptor_t cp = (c.dt == CUDA_R_32F) ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;
            report(std::string("7e ") + c.lbl, runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, cp, sc, false), c.dt,
                   "packed col-major, all four descriptors with this alignmentRequirement", true);
        }
    }

    printf("--- check 8: rank-0 operand (numModes = 0) ---\n");
    {
        TSpec s0({}, {}, {});
        Tsr A(s0, rng), B(colMajor({'m', 'k'}, {m, k}), rng), C(colMajor({'m', 'k'}, {m, k}), rng),
            D(colMajor({'m', 'k'}, {m, k}), rng, false, SENT);
        report("8a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "D = s*B + 0.5*C with s = operand A of numModes=0 (scalar), B rank-2 (37,53) packed");
    }
    {
        TSpec s0({}, {}, {});
        Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'm'}, {k, m}), rng), C(s0, rng), D(s0, rng, false, SENT);
        report("8b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "rank-0 output: D = sum_{m,k} A[m,k]*B[k,m] (full reduction to a scalar)");
    }
    {
        TSpec s0({}, {}, {});
        Tsr A(s0, rng), B(s0, rng), C(s0, rng), D(s0, rng, false, SENT);
        report("8c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "all four rank-0: D = a*b + 0.5*c");
    }
    {
        TSpec s0({}, {}, {});
        TSpec sB({'m', 'k'}, {m, k}, {2 * k + 3, 2}, CUDA_R_64F, 8);
        TSpec sD({'m', 'k'}, {m, k}, {2 * k + 1, 2}, CUDA_R_64F, 8);
        Tsr A(s0, rng), B(sB, rng), D(sD, rng, false, SENT), C(sD, rng);
        report("8d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc5, true), CUDA_R_64F,
               "rank-0 scalar x strided B (109,2) into strided D (107,2); beta=0.5");
    }

    printf("--- check 9: same mode twice in one tensor (A[i,i]) ---\n");
    {
        // diagonal of a (53,53) matrix with leading dimension 54: A[i,j] at i + 54 j; modes {i,i} -> A[i,i] at 55 i
        TSpec sA({'i', 'i'}, {k, k}, {1, k + 1}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(colMajor({'i', 'n'}, {k, n}), rng), C(colMajor({'n'}, {n}), rng),
            D(colMajor({'n'}, {n}), rng, false, SENT);
        report("9a", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "modeA = {i,i}, A ext (53,53) strides (1,54); D[n] = sum_i A[i,i]*B[i,n]; docs: mode at most once", true);
    }
    {
        TSpec sA({'i', 'i', 'n'}, {k, k, n}, {1, k + 1, k * (k + 1)}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(colMajor({'i', 'n'}, {k, n}), rng), C(colMajor({'n'}, {n}), rng),
            D(colMajor({'n'}, {n}), rng, false, SENT);
        report("9b", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "modeA = {i,i,n} (rank-3 A with duplicated contracted mode)", true);
    }
    {
        // does cuTENSOR sum the strides of a duplicated mode (A[i,i] at (2+54)i) or use only one occurrence?
        TSpec sA({'i', 'i'}, {k, k}, {2, k + 1}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(colMajor({'i', 'n'}, {k, n}), rng), C(colMajor({'n'}, {n}), rng),
            D(colMajor({'n'}, {n}), rng, false, SENT);
        report("9c", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "modeA = {i,i}, A strides (2,54): reference assumes offset = (2+54)*i (sum of the duplicated strides)", true);
    }
    {
        TSpec sA({'i', 'i'}, {k, k}, {3, 5}, CUDA_R_64F, 8);
        Tsr A(sA, rng), B(colMajor({'i', 'n'}, {k, n}), rng), C(colMajor({'n'}, {n}), rng),
            D(colMajor({'n'}, {n}), rng, false, SENT);
        report("9d", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, comp, sc0, true), CUDA_R_64F,
               "modeA = {i,i}, A strides (3,5): reference assumes offset = 8*i", true);
    }
}

// Isolated: over-declared alignment on a non-GEMM contraction (cuTENSOR's own kernels, not cuBLAS).
void checkAlignExtra(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize) {
    std::mt19937_64 rng(20261127);
    const int m = 37, k1 = 11, k2 = 13, n = 41, p = 7;
    Scalars sc = scalarsF64(1.0, 0.5);
    // packed rank-3 A[m,k1,k2] x B[k2,n,k1,p] -> D[m,n,p], all pointers offset by one f64 element (8-byte aligned)
    for (uint32_t al : {8u, 16u, 256u}) {
        TSpec sA = colMajor({'m', 'k', 'l'}, {m, k1, k2}, CUDA_R_64F, al);
        TSpec sB = colMajor({'l', 'n', 'k', 'p'}, {k2, n, k1, p}, CUDA_R_64F, al);
        TSpec sD = colMajor({'m', 'n', 'p'}, {m, n, p}, CUDA_R_64F, al);
        sA.baseOff = 1;
        sB.baseOff = 1;
        sD.baseOff = 1;
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
        char note[128];
        snprintf(note, sizeof note, "non-GEMM rank-3/rank-4 contraction; A, B, C, D base pointers at +8 bytes; alignmentRequirement=%u declared", al);
        report(std::string("7f align ") + std::to_string(al), runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_64F, sc, true),
               CUDA_R_64F, note);
    }
    // strides of 3 elements (24 B) in every operand with alignmentRequirement 256 declared
    for (uint32_t al : {8u, 256u}) {
        TSpec sA({'m', 'k', 'l'}, {m, k1, k2}, {3, 3 * m, 3 * m * k1}, CUDA_R_64F, al);
        TSpec sB({'l', 'n', 'k', 'p'}, {k2, n, k1, p}, {3, 3 * k2, 3 * k2 * n, 3 * k2 * n * k1}, CUDA_R_64F, al);
        TSpec sD({'m', 'n', 'p'}, {m, n, p}, {3, 3 * m, 3 * m * n}, CUDA_R_64F, al);
        Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
        char note[128];
        snprintf(note, sizeof note, "non-GEMM; all strides multiples of 3 elements (24 B); alignmentRequirement=%u declared", al);
        report(std::string("7g align ") + std::to_string(al), runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_64F, sc, true),
               CUDA_R_64F, note);
    }
}
