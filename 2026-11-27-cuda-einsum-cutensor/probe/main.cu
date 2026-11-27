// cuTENSOR probe driver: probe <version|strides|conj|dtype|perf|all> [check-filter]
#include "probe_common.h"

void checkStrides(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize);
void checkConj(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize);
void checkDtype(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize);
void checkPerf(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize, bool includeCorrectness);
void planSequence(cutensorHandle_t h, int n, char** sizes);
void checkAlignExtra(cutensorHandle_t h, cudaStream_t stream, void* ws, uint64_t wsSize);

static void printVersions(cutensorHandle_t h) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    size_t v = cutensorGetVersion(), cv = cutensorGetCudartVersion();
    printf("[19] cutensorGetVersion() = %zu (major.minor.patch = %zu.%zu.%zu)\n", v, v / 10000, (v / 100) % 100, v % 100);
    printf("[19] cutensorGetCudartVersion() = %zu (%zu.%zu)\n", cv, cv / 1000, (cv / 10) % 100);
    printf("[19] cudaRuntimeGetVersion() = %d, cudaDriverGetVersion() = %d\n", rt, drv);
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
    printf("[19] GPU = \"%s\", compute capability %d.%d, %d SMs, %.2f GiB, nominal SM clock %.0f MHz\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, clockKHz / 1000.0);
    cublasHandle_t cb;
    cublasCreate(&cb);
    int cv2 = 0;
    cublasGetVersion(cb, &cv2);
    printf("[19] cuBLAS version = %d\n", cv2);
    cublasDestroy(cb);
    fflush(stdout);
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "all";
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    cutensorHandle_t h = nullptr;
    cutensorStatus_t s = cutensorCreate(&h);
    if (!isok(s)) {
        fprintf(stderr, "cutensorCreate failed: status %d (%s)\n", (int)s, sname(s));
        return 1;
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const uint64_t WS = 128ull << 20;
    void* ws = nullptr;
    CUDA_CHECK(cudaMalloc(&ws, WS));
    if (cmd == "version" || cmd == "all") printVersions(h);
    if (cmd == "strides" || cmd == "all") checkStrides(h, stream, ws, WS);
    if (cmd == "conj" || cmd == "all") checkConj(h, stream, ws, WS);
    if (cmd == "dtype" || cmd == "all") checkDtype(h, stream, ws, WS);
    if (cmd == "perf") checkPerf(h, stream, ws, WS, false);
    if (cmd == "plan") planSequence(h, argc - 2, argv + 2);
    if (cmd == "align-extra") checkAlignExtra(h, stream, ws, WS);
    if (cmd == "perf-after-others") checkPerf(h, stream, ws, WS, true);
    if (cmd == "gemm-perf-only") {
        // skips check 20 (planning) so it can be run in isolation
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t ce = cudaGetLastError();
    printf("--- done (%s); last CUDA error: %s ---\n", cmd.c_str(), cudaGetErrorString(ce));
    fflush(stdout);
    return 0;
}
