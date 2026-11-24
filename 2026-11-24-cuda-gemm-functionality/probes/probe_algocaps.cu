// Which algorithms (if any) on this GPU advertise FP emulation support?
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static const int M = 512, N = 512, K = 512;

static void report(cublasLtHandle_t lt, const char* name, cublasComputeType_t ct,
                   cudaDataType_t dt, cudaDataType_t st) {
    int32_t ids[256];
    int nids = 0;
    cublasStatus_t s = cublasLtMatmulAlgoGetIds(lt, ct, st, dt, dt, dt, dt, 256, ids, &nids);
    printf("%-28s algo ids: status=%d count=%d\n", name, (int)s, nids);
    int emu = 0;
    for (int i = 0; i < nids; ++i) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(lt, ct, st, dt, dt, dt, dt, ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        int32_t cap = 0; size_t sz = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT,
                                              &cap, sizeof(cap), &sz) == CUBLAS_STATUS_SUCCESS && cap != 0)
            ++emu;
    }
    printf("%-28s algos advertising FP emulation: %d / %d\n", "", emu, nids);

    // what does the heuristic pick?
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, ct, st);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
    cublasLtMatrixLayoutCreate(&Ad, dt, M, K, M);
    cublasLtMatrixLayoutCreate(&Bd, dt, K, N, K);
    cublasLtMatrixLayoutCreate(&Cd, dt, M, N, M);
    cublasLtMatrixLayoutCreate(&Dd, dt, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 64ull << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur; int nres = 0;
    s = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &nres);
    int32_t hcap = -1; size_t hsz = 0;
    if (s == CUBLAS_STATUS_SUCCESS && nres > 0)
        cublasLtMatmulAlgoCapGetAttribute(&heur.algo, CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT,
                                          &hcap, sizeof(hcap), &hsz);
    printf("%-28s heuristic status=%d n=%d, chosen algo emu_cap=%d\n", "", (int)s, nres, hcap);
    printf("\n");
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
    cublasLtMatmulDescDestroy(op);
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    report(lt, "f32 compute 32F", CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F);
    report(lt, "f32 compute 16BFX9", CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUDA_R_32F, CUDA_R_32F);
    report(lt, "f64 compute 64F", CUBLAS_COMPUTE_64F, CUDA_R_64F, CUDA_R_64F);
    report(lt, "f64 compute FIXEDPOINT", CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F, CUDA_R_64F);
    cublasLtDestroy(lt);
    return 0;
}
