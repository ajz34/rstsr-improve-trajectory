// Check 18: f16 / bf16 / f32 with COMPUTE_DESC_32F and float scalars (CPU reference accumulates in float).
#include "probe_common.h"

void checkDtype(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize) {
    std::mt19937_64 rng(20261127);
    const int m = M0, k = K0, n = N0;
    struct DT { cudaDataType_t dt; const char* name; };
    const DT dts[3] = {{CUDA_R_16F, "f16"}, {CUDA_R_16BF, "bf16"}, {CUDA_R_32F, "f32"}};
    for (const DT& d : dts) {
        cudaDataType_t dt = d.dt;
        uint32_t al = (eszOf(dt) == 2) ? 2u : 8u;  // true base/stride alignment of the strided layout below
        Scalars sc0 = scalarsF32(1.0f, 0.0f), sc5 = scalarsF32(1.0f, 0.5f);
        printf("--- check 18: %s with CUTENSOR_COMPUTE_DESC_32F, float scalars ---\n", d.name);
        {
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            report(std::string("18-") + d.name + "-packed", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_32F, sc0, false),
                   dt, "packed col-major; beta=0; alpha/beta passed as float");
        }
        {
            TSpec sA({'m', 'k'}, {m, k}, {2 * k + 3, 2}, dt, al);
            TSpec sB({'k', 'n'}, {k, n}, {1, k + 5}, dt, al);
            TSpec sD({'m', 'n'}, {m, n}, {2 * n, 1}, dt, al);
            Tsr A(sA, rng), B(sB, rng), D(sD, rng, false), C(sD, rng);
            char note[256];
            snprintf(note, sizeof note, "strided: A (109,2) B (1,58) D (82,1), alignmentRequirement=%u; beta=0.5", al);
            report(std::string("18-") + d.name + "-strided", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_32F, sc5, false),
                   dt, note);
        }
        if (dt == CUDA_R_32F) {
            // TF32 and 3xTF32 compute types on f32 data (tensor cores; Ampere+)
            Tsr A(colMajor({'m', 'k'}, {m, k}, dt), rng), B(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C(colMajor({'m', 'n'}, {m, n}, dt), rng), D(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            reportTol("18-f32-TF32", runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_TF32, sc0, false), 1e-2,
                      "f32 data with COMPUTE_DESC_TF32 (10-bit mantissa products; float CPU reference; tolerance 1e-2 relative)");
            Tsr A2(colMajor({'m', 'k'}, {m, k}, dt), rng), B2(colMajor({'k', 'n'}, {k, n}, dt), rng),
                C2(colMajor({'m', 'n'}, {m, n}, dt), rng), D2(colMajor({'m', 'n'}, {m, n}, dt), rng, false);
            reportTol("18-f32-3xTF32", runTyped(h, stream, ws, wsSize, A2, ID, B2, ID, C2, ID, D2, CUTENSOR_COMPUTE_DESC_3XTF32, sc0, false), 1e-5,
                      "f32 data with COMPUTE_DESC_3XTF32 (float CPU reference; tolerance 1e-5 relative)");
        }
    }
    // f64 with COMPUTE_DESC_32F (listed in docs: CUDA_R_64F x3 with 32F compute, typeScalar f64) -- reduced precision
    {
        Scalars sc0 = scalarsF64(1.0, 0.0);
        Tsr A(colMajor({'m', 'k'}, {m, k}, CUDA_R_64F), rng), B(colMajor({'k', 'n'}, {k, n}, CUDA_R_64F), rng),
            C(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng), D(colMajor({'m', 'n'}, {m, n}, CUDA_R_64F), rng, false);
        CheckResult r = runTyped(h, stream, ws, wsSize, A, ID, B, ID, C, ID, D, CUTENSOR_COMPUTE_DESC_32F, sc0, false);
        // judge against f32 tolerance since the compute type is 32F
        if (!r.anyErr && r.haveResult) {
            double norm = r.maxRef > 0 ? r.maxAbs / r.maxRef : r.maxAbs;
            char buf[256];
            snprintf(buf, sizeof buf, "maxabs=%.3e maxrel=%.3e maxabs/max|ref|=%.3e", r.maxAbs, r.maxRel, norm);
            printf("[18-f64-32F] %s | %s | f64 data with COMPUTE_DESC_32F (docs list it; scalars stay double): %s\n",
                   norm <= tolOf(CUDA_R_32F) ? "ok (f32 tol)" : norm <= tolOf(CUDA_R_64F) ? "ok" : "FAIL", buf,
                   norm <= tolOf(CUDA_R_64F) ? "result is f64-accurate" : "result only f32-accurate");
        } else {
            report("18-f64-32F", r, CUDA_R_64F, "f64 data with COMPUTE_DESC_32F (docs list it; scalars stay double)");
        }
        fflush(stdout);
    }
}
