// Minimal cuTENSOR via pure dlopen/dlsym (no cutensor headers, no link-time dependency).
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef void* handle_t;   // cutensorHandle_t (opaque ptr)
typedef void* desc_t;     // cutensorTensorDescriptor_t (opaque ptr)
typedef void* op_t;       // cutensorOperationDescriptor_t (opaque ptr)
typedef void* compute_t;  // cutensorComputeDescriptor_t (opaque ptr)
typedef int status_t;     // cutensorStatus_t
typedef int op_tor;       // cutensorOperator_t

typedef status_t (*fn_create)(handle_t*);
typedef status_t (*fn_mdesc)(const handle_t, desc_t*, uint32_t, const int64_t[], const int64_t[], int, uint32_t);
typedef status_t (*fn_contraction)(const handle_t, op_t*, const desc_t, const int32_t[], op_tor,
                                   const desc_t, const int32_t[], op_tor,
                                   const desc_t, const int32_t[], op_tor,
                                   const desc_t, const int32_t[], compute_t);
typedef uint64_t (*fn_version)();

#define CUTENSOR_OP_IDENTITY 1
#define CUDA_R_64F 1

int main() {
    void* lib = dlopen("libcutensor.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    fn_version get_version = (fn_version)dlsym(lib, "cutensorGetVersion");
    printf("version: %lu\n", (unsigned long)get_version());
    fn_create create = (fn_create)dlsym(lib, "cutensorCreate");
    fn_mdesc mdesc = (fn_mdesc)dlsym(lib, "cutensorCreateTensorDescriptor");
    fn_contraction contraction = (fn_contraction)dlsym(lib, "cutensorCreateContraction");
    compute_t compute = *(compute_t*)dlsym(lib, "CUTENSOR_COMPUTE_DESC_64F");  // deref: dlsym gives &global
    handle_t h;
    printf("cutensorCreate: %d\n", create(&h));
    desc_t da, db, dc, dd;
    int64_t ea[2] = {7, 13}, sa[2] = {13, 1};
    int64_t eb[2] = {13, 5}, sb[2] = {5, 1};
    int64_t ec[2] = {7, 5}, sc[2] = {5, 1};
    printf("descA: %d\n", mdesc(h, &da, 2, ea, sa, CUDA_R_64F, 256));
    printf("descB: %d\n", mdesc(h, &db, 2, eb, sb, CUDA_R_64F, 256));
    printf("descC: %d\n", mdesc(h, &dc, 2, ec, sc, CUDA_R_64F, 256));
    printf("descD: %d\n", mdesc(h, &dd, 2, ec, sc, CUDA_R_64F, 256));
    int32_t ma[2] = {0, 1}, mb[2] = {1, 2}, mc[2] = {0, 2};
    op_t op;
    printf("CreateContraction: %d\n",
        contraction(h, &op, da, ma, CUTENSOR_OP_IDENTITY, db, mb, CUTENSOR_OP_IDENTITY,
                    dc, mc, CUTENSOR_OP_IDENTITY, dd, mc, compute));
    return 0;
}
