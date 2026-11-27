// Minimal debug: D = alpha*A*B + beta*C, packed f64, print first elements. Usage: debug_beta
#include "probe_common.h"

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    cutensorHandle_t h;
    cutensorCreate(&h);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, 128ull << 20));
    std::mt19937_64 rng(1);
    const int m = argc > 1 ? atoi(argv[1]) : 4, k = argc > 2 ? atoi(argv[2]) : 3, n = argc > 3 ? atoi(argv[3]) : 5;
    printf("m=%d k=%d n=%d\n", m, k, n);
    Tsr A(colMajor({'m', 'k'}, {m, k}), rng), B(colMajor({'k', 'n'}, {k, n}), rng), C(colMajor({'m', 'n'}, {m, n}), rng),
        D(colMajor({'m', 'n'}, {m, n}), rng, false, W(-7.5e300));
    // same descriptor object for C and D vs. separate: try both
    for (int variant = 0; variant < 3; ++variant) {
        for (size_t i = 0; i < D.buf->n; ++i) writeElem(D.buf->h0.data(), CUDA_R_64F, (int64_t)i, W(-7.5e300));
        CUDA_CHECK(cudaMemcpy(D.buf->d, D.buf->h0.data(), D.buf->n * 8, cudaMemcpyHostToDevice));
        cutensorTensorDescriptor_t dA, dB, dC, dD;
        cutensorCreateTensorDescriptor(h, &dA, 2, A.s.extent.data(), A.s.stride.data(), CUDA_R_64F, 256);
        cutensorCreateTensorDescriptor(h, &dB, 2, B.s.extent.data(), B.s.stride.data(), CUDA_R_64F, 256);
        cutensorCreateTensorDescriptor(h, &dD, 2, D.s.extent.data(), D.s.stride.data(), CUDA_R_64F, 256);
        if (variant == 0) dC = dD;  // same object
        else cutensorCreateTensorDescriptor(h, &dC, 2, C.s.extent.data(), C.s.stride.data(), CUDA_R_64F, 256);
        cutensorOperationDescriptor_t op;
        cutensorStatus_t s = cutensorCreateContraction(h, &op, dA, A.s.modes.data(), CUTENSOR_OP_IDENTITY, dB, B.s.modes.data(),
                                                       CUTENSOR_OP_IDENTITY, dC, C.s.modes.data(), CUTENSOR_OP_IDENTITY, dD,
                                                       D.s.modes.data(), CUTENSOR_COMPUTE_DESC_64F);
        printf("variant %d: CreateContraction status %d\n", variant, (int)s);
        cutensorPlanPreference_t pref;
        cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
        uint64_t wse = 0;
        cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &wse);
        cutensorPlan_t plan;
        s = cutensorCreatePlan(h, &plan, op, pref, 128ull << 20);
        printf("  CreatePlan status %d, wsEstimate %llu\n", (int)s, (unsigned long long)wse);
        double alpha = 1.0, beta = 0.5;
        if (variant == 2) {
            // pass C pointer = D pointer (aliasing) with D pre-filled by C's content
            CUDA_CHECK(cudaMemcpy(D.buf->d, C.buf->d, D.buf->n * 8, cudaMemcpyDeviceToDevice));
        }
        s = cutensorContract(h, plan, &alpha, A.buf->dptr(), B.buf->dptr(), &beta, variant == 2 ? D.buf->dptr() : C.buf->dptr(),
                             D.buf->dptr(), ws, 128ull << 20, stream);
        cudaError_t ce = cudaStreamSynchronize(stream);
        printf("  Contract status %d (%s), cuda sync %s\n", (int)s, sname(s), cudaGetErrorString(ce));
        D.download();
        // reference
        auto ext = modeExtents({&A.s, &B.s, &C.s, &D.s});
        std::vector<uint8_t> href = variant == 2 ? C.buf->h0 : D.buf->h0;
        std::vector<uint8_t> hrefNoBeta = href;
        refContract<double>(A.buf->h0.data(), A.s, false, B.buf->h0.data(), B.s, false, 1.0, 0.5, C.buf->h0.data(), C.s, false, href.data(),
                            D.s, ext);
        refContract<double>(A.buf->h0.data(), A.s, false, B.buf->h0.data(), B.s, false, 1.0, 0.0, C.buf->h0.data(), C.s, false,
                            hrefNoBeta.data(), D.s, ext);
        double e1 = 0, e0 = 0;
        for (int64_t o : touchedOffsets(D.s, ext)) {
            W g = readElem(D.buf->h.data(), CUDA_R_64F, o), r = readElem(href.data(), CUDA_R_64F, o), r0 = readElem(hrefNoBeta.data(), CUDA_R_64F, o);
            e1 = std::max(e1, std::abs(g - r));
            e0 = std::max(e0, std::abs(g - r0));
        }
        printf("  maxerr vs A*B+0.5C = %.3e ; maxerr vs A*B = %.3e\n", e1, e0);
        for (int i = 0; i < 3; ++i) {
            W g = readElem(D.buf->h.data(), CUDA_R_64F, i), r = readElem(href.data(), CUDA_R_64F, i),
              r0 = readElem(hrefNoBeta.data(), CUDA_R_64F, i), c = readElem(C.buf->h0.data(), CUDA_R_64F, i);
            printf("  D[%d] gpu=%.6f  ref(A*B+0.5C)=%.6f  ref(A*B)=%.6f  C=%.6f\n", i, g.real(), r.real(), r0.real(), c.real());
        }
        cutensorDestroyPlan(plan);
        cutensorDestroyPlanPreference(pref);
        cutensorDestroyOperationDescriptor(op);
        cutensorDestroyTensorDescriptor(dA);
        cutensorDestroyTensorDescriptor(dB);
        cutensorDestroyTensorDescriptor(dD);
        if (variant != 0) cutensorDestroyTensorDescriptor(dC);
        printf("\n");
    }
    return 0;
}
