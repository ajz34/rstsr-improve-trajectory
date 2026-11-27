// facts.cpp -- cuTENSOR elementwise functionality probe.
//
// Groups:
//   g1  dtype x operator matrix (same-type storage, documented table)
//   g2  dtype mixing and compute-type variations (promotion question)
//   g3  layout handling: transposed, windowed, stepped, broadcast, negative, strided output
//   g4  composition: unary-via-binary, binary-via-trinary, fused ternary, scalars, JIT, plan cost
#include "common.h"

static cutensorHandle_t H = nullptr;

static std::string ab(cutensorStatus_t s) { return s == CUTENSOR_STATUS_SUCCESS ? "OK" : std::string(st_name(s)); }

struct Outcome {
    std::string stages;
    std::string exec;
    std::string verify;
    cudaDataType_t scalar = CUDA_R_32F;
    bool ok = false;
};

static std::string stage_str(const Plan& p) {
    return "create=" + ab(p.s_op) + " pref=" + ab(p.s_pref) + " est=" + ab(p.s_est) + " plan=" + ab(p.s_plan);
}

// ------------------------------------------------------------------------------------------- g1
static void g1_one_unary(const DType& dt, cutensorOperator_t op, const char* opname) {
    const int64_t n = 35;
    TensorSpec A{dt.v, {7, 5}, {5, 1}, 0};
    TensorSpec CD{dt.v, {7, 5}, {5, 1}, 0};
    bool pos = (op == CUTENSOR_OP_SQRT || op == CUTENSOR_OP_RCP || op == CUTENSOR_OP_LOG);
    auto vals = make_vals((size_t)n, 11, dt.is_complex, pos);
    TV ta{A, vals};
    std::vector<uint8_t> hb = stage(ta);
    DBuf da(hb.size()), dout((size_t)n * dt.bytes);
    da.upload(hb.data(), hb.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    Plan p = make_binary_plan(H, A, op, CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, default_compute(dt.v));
    Outcome o;
    o.stages = stage_str(p);
    o.scalar = p.scalar;
    if (!p.ok) {
        emit("G1", std::string(dt.name) + " unary " + opname, o.stages, "scalar=" + dt_name(p.scalar));
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 0.0, 0.0);
    cutensorStatus_t e = exec_binary(H, p, s, da.p, dout.p, dout.p);
    o.exec = ab(e);
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) o.exec += std::string(" cuda=") + cudaGetErrorString(ce);

    TV to{CD, {}};
    auto out = fetch(to, dout.p);
    int bad = 0;
    double tol = tol_for(dt.v);
    for (int64_t i = 0; i < n; ++i) {
        cd want = ref_unary(op, vals[(size_t)i]);
        if (!close(out[(size_t)i], want, tol, tol)) ++bad;
    }
    char b[96];
    snprintf(b, sizeof b, "verify=%s(%d/%lld) scalar=%s", bad ? "FAIL" : "OK", bad, (long long)n, dt_name(p.scalar).c_str());
    emit("G1", std::string(dt.name) + " unary " + opname, o.stages + " exec=" + o.exec, b);
}

static void g1_one_binary(const DType& dt, cutensorOperator_t op, const char* opname, bool via_binary_api) {
    const int64_t n = 35;
    TensorSpec A{dt.v, {7, 5}, {5, 1}, 0};
    TensorSpec B{dt.v, {7, 5}, {5, 1}, 0};
    TensorSpec CD{dt.v, {7, 5}, {5, 1}, 0};
    auto av = make_vals((size_t)n, 21, dt.is_complex);
    auto bv = make_vals((size_t)n, 22, dt.is_complex);
    TV ta{A, av}, tb{B, bv};
    std::vector<uint8_t> hba = stage(ta), hbb = stage(tb);
    DBuf da(hba.size()), db(hbb.size()), dout((size_t)n * dt.bytes);
    da.upload(hba.data(), hba.size());
    db.upload(hbb.data(), hbb.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    Plan p = via_binary_api
        ? make_binary_plan(H, A, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, op, default_compute(dt.v))
        : make_trinary_plan(H, A, CUTENSOR_OP_IDENTITY, B, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, op,
                            CUTENSOR_OP_ADD, default_compute(dt.v));
    Outcome o;
    o.stages = stage_str(p);
    o.scalar = p.scalar;
    if (!p.ok) {
        emit("G1", std::string(dt.name) + " " + (via_binary_api ? "bin-api " : "trin ") + opname, o.stages,
             "scalar=" + dt_name(p.scalar));
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 1.0, via_binary_api ? 1.0 : 0.0);
    cutensorStatus_t e = via_binary_api ? exec_binary(H, p, s, da.p, db.p, dout.p)
                                        : exec_trinary(H, p, s, da.p, db.p, dout.p, dout.p);
    o.exec = ab(e);
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) o.exec += std::string(" cuda=") + cudaGetErrorString(ce);

    TV to{CD, {}};
    auto out = fetch(to, dout.p);
    int bad = 0;
    double tol = tol_for(dt.v);
    for (int64_t i = 0; i < n; ++i) {
        cd want = ref_binary(op, av[(size_t)i], bv[(size_t)i]);
        if (!close(out[(size_t)i], want, tol, tol)) ++bad;
    }
    char b[96];
    snprintf(b, sizeof b, "verify=%s(%d/%lld) scalar=%s", bad ? "FAIL" : "OK", bad, (long long)n, dt_name(p.scalar).c_str());
    emit("G1", std::string(dt.name) + " " + (via_binary_api ? "bin-api " : "trin ") + opname, o.stages + " exec=" + o.exec, b);
}

static void g1_dtypes() {
    const DType* dts[] = {&DT_R16F, &DT_R16BF, &DT_R32F, &DT_R64F, &DT_C32F, &DT_C64F, &DT_R32I, &DT_R64I, &DT_R8U};
    struct U { cutensorOperator_t op; const char* n; };
    struct B { cutensorOperator_t op; const char* n; };
    U unary[] = {{CUTENSOR_OP_IDENTITY, "IDENTITY"}, {CUTENSOR_OP_NEG, "NEG"},    {CUTENSOR_OP_ABS, "ABS"},
                 {CUTENSOR_OP_CONJ, "CONJ"},         {CUTENSOR_OP_EXP, "EXP"},    {CUTENSOR_OP_SQRT, "SQRT"},
                 {CUTENSOR_OP_RCP, "RCP"},           {CUTENSOR_OP_CEIL, "CEIL"},  {CUTENSOR_OP_SIN, "SIN"}};
    B binary[] = {{CUTENSOR_OP_ADD, "ADD"}, {CUTENSOR_OP_MUL, "MUL"}, {CUTENSOR_OP_MAX, "MAX"}, {CUTENSOR_OP_MIN, "MIN"}};
    for (const DType* dt : dts) {
        for (auto& u : unary) g1_one_unary(*dt, u.op, u.n);
        for (auto& b : binary) {
            g1_one_binary(*dt, b.op, b.n, false);
            g1_one_binary(*dt, b.op, b.n, true);
        }
    }
}

// ------------------------------------------------------------------------------------------- g2
// One dtype-mixing cell: A/B storage types and the C/D type, compute type, operation.
static void g2_cell(const char* tag, const DType& dA, const DType& dB, const DType& dCD, cutensorComputeDescriptor_t comp,
                    bool trinary, cutensorOperator_t op = CUTENSOR_OP_ADD) {
    const int64_t n = 35;
    TensorSpec A{dA.v, {7, 5}, {5, 1}, 0};
    TensorSpec B{dB.v, {7, 5}, {5, 1}, 0};
    TensorSpec CD{dCD.v, {7, 5}, {5, 1}, 0};
    auto av = make_vals((size_t)n, 31, dA.is_complex);
    auto bv = make_vals((size_t)n, 32, dB.is_complex);
    TV ta{A, av}, tb{B, bv};
    std::vector<uint8_t> hba = stage(ta), hbb = stage(tb);
    DBuf da(hba.size()), db(hbb.size()), dout((size_t)n * dCD.bytes);
    da.upload(hba.data(), hba.size());
    db.upload(hbb.data(), hbb.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    Plan p = trinary ? make_trinary_plan(H, A, CUTENSOR_OP_IDENTITY, B, CUTENSOR_OP_IDENTITY, CD,
                                          CUTENSOR_OP_IDENTITY, op, CUTENSOR_OP_ADD, comp)
                     : make_binary_plan(H, A, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, op, comp);
    std::string head = std::string(tag) + " A=" + dA.name + " B=" + (trinary ? dB.name : "-") + " CD=" + dCD.name +
                       " comp=" + comp_name(comp);
    if (!p.ok) {
        emit("G2", head, stage_str(p), "scalar=" + dt_name(p.scalar));
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
    cutensorStatus_t e = trinary ? exec_trinary(H, p, s, da.p, db.p, dout.p, dout.p)
                                 : exec_binary(H, p, s, da.p, db.p, dout.p);
    cudaDeviceSynchronize();
    TV to{CD, {}};
    auto out = fetch(to, dout.p);
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        cd a_stored = decode(encode({av[(size_t)i]}, dA.v).data(), 1, dA.v)[0];
        cd b_stored = decode(encode({bv[(size_t)i]}, dB.v).data(), 1, dB.v)[0];
        cd sum = trinary ? ref_binary(op, a_stored, b_stored) : a_stored;
        cd want = decode(encode({sum}, dCD.v).data(), 1, dCD.v)[0];
        if (!close(out[(size_t)i], want, tol_for(dCD.v), tol_for(dCD.v))) ++bad;
    }
    char b[128];
    snprintf(b, sizeof b, "exec=%s verify=%s(%d/%lld) scalar=%s", ab(e).c_str(), bad ? "FAIL" : "OK", bad, (long long)n,
             dt_name(p.scalar).c_str());
    emit("G2", head, stage_str(p), b);
}

static void g2_promotion() {
    // unary A -> CD (documented: CD may be narrower; is the reverse accepted?)
    g2_cell("un-a", DT_R64F, DT_R64F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_R32F, DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_R16F, DT_R16F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, false);
    g2_cell("un-a", DT_R32F, DT_R32F, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, false);
    g2_cell("un-a", DT_C64F, DT_C64F, DT_C32F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_C32F, DT_C32F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_R64F, DT_R64F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_C64F, DT_C64F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, false);
    g2_cell("un-a", DT_R32I, DT_R32I, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, false);
    g2_cell("un-a", DT_R32F, DT_R32F, DT_R32I, CUTENSOR_COMPUTE_DESC_32F, false);

    // trinary: A, B, CD
    g2_cell("trin", DT_R32F, DT_R64F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_R64F, DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_R32F, DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_R64F, DT_R64F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_C64F, DT_R64F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_R64F, DT_C64F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_C32F, DT_C64F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("trin", DT_R16F, DT_R16F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, true);
    g2_cell("trin", DT_R32F, DT_R32F, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, true);
    g2_cell("trin", DT_R32I, DT_R32I, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, true);

    // compute-type variations with same-type storage
    g2_cell("comp", DT_R32F, DT_R32F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F, true);
    g2_cell("comp", DT_R64F, DT_R64F, DT_R64F, CUTENSOR_COMPUTE_DESC_32F, true);
    g2_cell("comp", DT_R16F, DT_R16F, DT_R16F, CUTENSOR_COMPUTE_DESC_16F, true);
    g2_cell("comp", DT_R16BF, DT_R16BF, DT_R16BF, CUTENSOR_COMPUTE_DESC_16BF, true);
}

// ----------------------------------------------------------------------------------------- g2c
// Full storage-pair matrix for the binary entry (copy form), rounded-input reference.
static void g2c_matrix() {
    const DType* dts[] = {&DT_R16F, &DT_R16BF, &DT_R32F, &DT_R64F, &DT_C32F, &DT_C64F};
    for (const DType* da : dts) {
        for (const DType* dc : dts) {
            int aw = (da->v == CUDA_R_64F || da->v == CUDA_C_64F) ? 64 : 32;
            int cw = (dc->v == CUDA_R_64F || dc->v == CUDA_C_64F) ? 64 : 32;
            cutensorComputeDescriptor_t comp = std::max(aw, cw) == 64 ? CUTENSOR_COMPUTE_DESC_64F
                                                                      : CUTENSOR_COMPUTE_DESC_32F;
            const int64_t n = 35;
            TensorSpec A{da->v, {7, 5}, {5, 1}, 0};
            TensorSpec CD{dc->v, {7, 5}, {5, 1}, 0};
            auto av = make_vals((size_t)n, 71, da->is_complex);
            std::vector<uint8_t> hba = encode(av, da->v);
            DBuf buf(hba.size()), out((size_t)n * dc->bytes);
            buf.upload(hba.data(), hba.size());
            cudaMemset(out.p, 0xFF, out.bytes);
            Plan p = make_binary_plan(H, A, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, comp);
            std::string head = std::string(da->name) + "->" + dc->name + " " + comp_name(comp);
            if (!p.ok) {
                emit("G2C", head, "plan=FAIL", ab(p.s_plan));
                continue;
            }
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 0.0, 0.0);
            cutensorStatus_t e = exec_binary(H, p, s, buf.p, out.p, out.p);
            cudaError_t ce = cudaDeviceSynchronize();
            TV t{CD, {}};
            auto got = fetch(t, out.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) {
                cd a_stored = decode(encode({av[(size_t)i]}, da->v).data(), 1, da->v)[0];
                cd want = decode(encode({a_stored}, dc->v).data(), 1, dc->v)[0];
                if (!close(got[(size_t)i], want, tol_for(dc->v), tol_for(dc->v))) ++bad;
            }
            char b[128];
            snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
                     bad ? "FAIL" : "OK", bad);
            emit("G2C", head, "plan=OK", b);
        }
    }
}

// ----------------------------------------------------------------------------------------- g2b
// Mixed A vs C/D in the binary entry, with raw dumps of the widening cases.
static void g2b_one(const char* tag, const DType& dA, const DType& dCD, cutensorComputeDescriptor_t comp,
                    bool with_C, bool dump) {
    const int64_t n = 35;
    TensorSpec A{dA.v, {7, 5}, {5, 1}, 0};
    TensorSpec CD{dCD.v, {7, 5}, {5, 1}, 0};
    auto av = make_vals((size_t)n, 61, dA.is_complex);
    auto bv = make_vals((size_t)n, 62, dCD.is_complex);
    std::vector<uint8_t> hba = encode(av, dA.v), hbb = encode(bv, dCD.v);
    DBuf da(hba.size()), dbc(hbb.size()), dout((size_t)n * dCD.bytes);
    da.upload(hba.data(), hba.size());
    dbc.upload(hbb.data(), hbb.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    Plan p = with_C ? make_binary_plan(H, A, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, comp)
                    : make_binary_plan(H, A, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, comp);
    std::string head = std::string(tag) + " A=" + dA.name + " CD=" + dCD.name + " comp=" + comp_name(comp) +
                       (with_C ? " (C read)" : " (copy)");
    if (!p.ok) {
        emit("G2B", head, stage_str(p), "");
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 0.0, with_C ? 1.0 : 0.0);
    cutensorStatus_t e = exec_binary(H, p, s, da.p, with_C ? dbc.p : dout.p, dout.p);
    cudaDeviceSynchronize();
    TV t{CD, {}};
    auto out = fetch(t, dout.p);
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        cd a_stored = decode(encode({av[(size_t)i]}, dA.v).data(), 1, dA.v)[0];
        cd b_stored = decode(encode({bv[(size_t)i]}, dCD.v).data(), 1, dCD.v)[0];
        cd sum = a_stored + (with_C ? b_stored : cd(0, 0));
        cd want = decode(encode({sum}, dCD.v).data(), 1, dCD.v)[0];
        if (!close(out[(size_t)i], want, tol_for(dCD.v), tol_for(dCD.v))) ++bad;
    }
    char b[200];
    snprintf(b, sizeof b, "exec=%s verify=%s(%d/%lld)", ab(e).c_str(), bad ? "FAIL" : "OK", bad, (long long)n);
    emit("G2B", head, stage_str(p), b);
    if (dump && bad) {
        for (int64_t i = 0; i < 2; ++i) {
            cd a_stored = decode(encode({av[(size_t)i]}, dA.v).data(), 1, dA.v)[0];
            cd b_stored = decode(encode({bv[(size_t)i]}, dCD.v).data(), 1, dCD.v)[0];
            cd want = decode(encode({a_stored + (with_C ? b_stored : cd(0, 0))}, dCD.v).data(), 1, dCD.v)[0];
            printf("    i=%lld got=(%.9g,%.9g) want=(%.9g,%.9g)  raw=", (long long)i, out[(size_t)i].real(),
                   out[(size_t)i].imag(), want.real(), want.imag());
            std::vector<uint8_t> raw = encode({out[(size_t)i]}, dCD.v);
            for (int k = 0; k < std::min<int>(16, dCD.bytes); ++k) printf("%02x", raw[(size_t)k]);
            if (dCD.v == CUDA_C_64F || dCD.v == CUDA_R_64F) {
                printf(" as_f64=%.9g", ((double*)raw.data())[0]);
            }
            if (dCD.v == CUDA_C_32F || dCD.v == CUDA_R_32F) printf(" as_f32=%.9g", ((float*)raw.data())[0]);
            printf("\n");
        }
    }
}

static void g2b_mix() {
    // documented downcast rows, copy and C-read forms
    g2b_one("down", DT_R32F, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, false, true);
    g2b_one("down", DT_R64F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F, false, true);
    g2b_one("down", DT_C64F, DT_C32F, CUTENSOR_COMPUTE_DESC_64F, false, true);
    g2b_one("down", DT_R32F, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, true, true);
    g2b_one("down", DT_R64F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F, true, true);
    // widening: accepted by CreatePlan?
    g2b_one("up", DT_R16F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, false, true);
    g2b_one("up", DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, false, true);
    g2b_one("up", DT_R16F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, false, true);
    g2b_one("up", DT_C32F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, false, true);
    g2b_one("up", DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_32F, false, true);
    // C-read with mixed types
    g2b_one("mix", DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, true, true);
    g2b_one("mix", DT_R16F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, true, true);
    // real/complex mixing
    g2b_one("mix", DT_R64F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, false, false);
    g2b_one("mix", DT_C64F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, false, false);
}

// ----------------------------------------------------------------------------------------- g2d
// Trinary entry with A/B at one width and C/D at another.
static void g2d_one(const char* tag, const DType& dAB, const DType& dCD, cutensorComputeDescriptor_t comp) {
    const int64_t n = 35;
    TensorSpec AB{dAB.v, {7, 5}, {5, 1}, 0};
    TensorSpec CD{dCD.v, {7, 5}, {5, 1}, 0};
    auto av = make_vals((size_t)n, 81, dAB.is_complex);
    auto bv = make_vals((size_t)n, 82, dAB.is_complex);
    std::vector<uint8_t> ba = encode(av, dAB.v), bb = encode(bv, dAB.v);
    DBuf da(ba.size()), db(bb.size()), dd((size_t)n * dCD.bytes);
    da.upload(ba.data(), ba.size());
    db.upload(bb.data(), bb.size());
    cudaMemset(dd.p, 0, dd.bytes);
    Plan p = make_trinary_plan(H, AB, CUTENSOR_OP_IDENTITY, AB, CUTENSOR_OP_IDENTITY, CD, CUTENSOR_OP_IDENTITY,
                               CUTENSOR_OP_MUL, CUTENSOR_OP_ADD, comp);
    std::string head = std::string(tag) + " AB=" + dAB.name + " CD=" + dCD.name + " " + comp_name(comp);
    if (!p.ok) {
        emit("G2D", head, stage_str(p), "");
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
    cutensorStatus_t e = exec_trinary(H, p, s, da.p, db.p, dd.p, dd.p);
    cudaError_t ce = cudaDeviceSynchronize();
    TV t{CD, {}};
    auto out = fetch(t, dd.p);
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        cd a = decode(encode({av[(size_t)i]}, dAB.v).data(), 1, dAB.v)[0];
        cd b = decode(encode({bv[(size_t)i]}, dAB.v).data(), 1, dAB.v)[0];
        cd want = decode(encode({a * b}, dCD.v).data(), 1, dCD.v)[0];
        if (!close(out[(size_t)i], want, tol_for(dCD.v), tol_for(dCD.v))) ++bad;
    }
    char b[128];
    snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
             bad ? "FAIL" : "OK", bad);
    emit("G2D", head, stage_str(p), b);
}

static void g2d_trinary() {
    g2d_one("same", DT_R32F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F);
    g2d_one("up", DT_R32F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F);
    g2d_one("down", DT_R64F, DT_R32F, CUTENSOR_COMPUTE_DESC_64F);
    g2d_one("up", DT_R16F, DT_R32F, CUTENSOR_COMPUTE_DESC_32F);
    g2d_one("down", DT_R32F, DT_R16F, CUTENSOR_COMPUTE_DESC_32F);
    g2d_one("up", DT_C32F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F);
    g2d_one("bad", DT_R16F, DT_R64F, CUTENSOR_COMPUTE_DESC_64F);
    g2d_one("bad", DT_R16BF, DT_R32F, CUTENSOR_COMPUTE_DESC_32F);
    g2d_one("bad", DT_R32F, DT_C64F, CUTENSOR_COMPUTE_DESC_64F);
    g2d_one("bad", DT_R32I, DT_R32I, CUTENSOR_COMPUTE_DESC_32F);
}

// ------------------------------------------------------------------------------------------- g3
struct LSpec {
    std::vector<int64_t> ext, str;
    int64_t off = 0;
};

static TensorSpec spec_of(const DType& dt, const LSpec& l) { return TensorSpec{dt.v, l.ext, l.str, l.off}; }

static void g3_cell(const char* id, const DType& dt, const LSpec& la, const LSpec& lb, const LSpec& lout,
                    cutensorOperator_t op, bool unary) {
    auto out_ext = lout.ext;
    int64_t n = 1;
    for (auto e : out_ext) n *= e;
    TV ta{spec_of(dt, la), {}}, tb{spec_of(dt, lb), {}}, tout{spec_of(dt, lout), {}};
    int64_t na = 1;
    for (auto e : la.ext) na *= e;
    int64_t nb = 1;
    for (auto e : lb.ext) nb *= e;
    ta.logical = make_vals((size_t)na, 41, dt.is_complex, op == CUTENSOR_OP_EXP);
    tb.logical = make_vals((size_t)nb, 42, dt.is_complex, op == CUTENSOR_OP_EXP || unary);
    TV ttout{tout.spec, {}};
    ttout.spec.off = 0;

    std::vector<uint8_t> hba = stage(ta), hbb = stage(tb);
    std::vector<cd> pa = placed_logical(ta, hba), pb = placed_logical(tb, hbb);

    // cuTENSOR descriptors carry no offset: the caller adds it to the pointer.
    TensorSpec sa = ta.spec, sb = tb.spec, so = tout.spec;
    sa.off = sb.off = so.off = 0;
    size_t out_bytes = (size_t)std::max<int64_t>(tout.spec.extent_elems(), 1) * (size_t)dt.bytes;
    DBuf da(hba.size()), db(hbb.size()), dout(out_bytes);
    da.upload(hba.data(), hba.size());
    db.upload(hbb.data(), hbb.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    Plan p = unary ? make_binary_plan(H, sa, op, so, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, default_compute(dt.v))
                   : make_trinary_plan(H, sa, CUTENSOR_OP_IDENTITY, sb, CUTENSOR_OP_IDENTITY, so,
                                       CUTENSOR_OP_IDENTITY, op, CUTENSOR_OP_ADD, default_compute(dt.v));
    if (!p.ok) {
        emit("G3", std::string(id) + " " + spec_name(ta.spec) + " | " + spec_name(tout.spec), stage_str(p), "");
        return;
    }
    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
    void* pa_dev = (char*)da.p + la.off * dt.bytes;
    void* pb_dev = (char*)db.p + lb.off * dt.bytes;
    void* po_dev = (char*)dout.p + lout.off * dt.bytes;
    cutensorStatus_t e = unary ? exec_binary(H, p, s, pa_dev, po_dev, po_dev)
                               : exec_trinary(H, p, s, pa_dev, pb_dev, po_dev, po_dev);
    cudaError_t ce = cudaDeviceSynchronize();

    auto out = fetch(ttout, po_dev);
    int bad = 0;
    double tol = tol_for(dt.v);
    for (int64_t i = 0; i < n; ++i) {
        cd a = pa[(size_t)bcast_index(i, out_ext, la.ext)];
        cd want = unary ? ref_unary(op, a) : ref_binary(op, a, pb[(size_t)bcast_index(i, out_ext, lb.ext)]);
        if (!close(out[(size_t)i], want, tol, tol)) ++bad;
    }
    if (bad && getenv("G3DBG")) {
        int shown = 0;
        for (int64_t i = 0; i < n && shown < 10; ++i) {
            cd a = pa[(size_t)bcast_index(i, out_ext, la.ext)];
            cd want = unary ? ref_unary(op, a) : ref_binary(op, a, pb[(size_t)bcast_index(i, out_ext, lb.ext)]);
            if (close(out[(size_t)i], want, tol, tol)) continue;
            int64_t r = i / out_ext[1], c = i % out_ext[1];
            printf("  dbg (r=%lld,c=%lld) got=%.6f want=%.6f\n", (long long)r, (long long)c,
                   out[(size_t)i].real(), want.real());
            ++shown;
        }
    }
    char b[160];
    snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d/%lld)", ab(e).c_str(), cudaGetErrorString(ce),
             bad ? "FAIL" : "OK", bad, (long long)n);
    emit("G3", std::string(id) + " " + spec_name(ta.spec) + " | " + spec_name(tout.spec), stage_str(p), b);
}

static void g3_views() {
    const int64_t R = 37, C = 53;
    LSpec contig{{R, C}, {C, 1}, 0};
    LSpec trans{{R, C}, {1, C}, 0};               // logical [R,C] of a [C,R] arena
    LSpec window{{R, C}, {C + 20, 1}, 10 * (C + 20) + 10};
    LSpec step{{R, C}, {4 * C, 2}, 0};            // arena [2R,2C], sliced ::2
    LSpec bcast_omit{{R}, {1}, 0};                // broadcast over the second axis
    LSpec bcast_zero{{R, C}, {1, 0}, 0};          // stride-0 second axis
    LSpec neg{{R, C}, {-C, 1}, (R - 1) * C};      // reversed rows
    LSpec out_trans{{R, C}, {1, R}, 0};
    LSpec out_window{{R, C}, {C + 20, 1}, 10 * (C + 20) + 10};
    LSpec out_step{{R, C}, {4 * C, 2}, 0};

    g3_cell("contig", DT_R64F, contig, contig, contig, CUTENSOR_OP_ADD, false);
    g3_cell("A.T", DT_R64F, trans, contig, contig, CUTENSOR_OP_ADD, false);
    g3_cell("B.T", DT_R64F, contig, trans, contig, CUTENSOR_OP_ADD, false);
    g3_cell("both.T", DT_R64F, trans, trans, contig, CUTENSOR_OP_ADD, false);
    g3_cell("window", DT_R64F, window, window, contig, CUTENSOR_OP_ADD, false);
    g3_cell("step", DT_R64F, step, step, contig, CUTENSOR_OP_ADD, false);
    g3_cell("bcast-omit", DT_R64F, contig, bcast_omit, contig, CUTENSOR_OP_ADD, false);
    g3_cell("bcast-zero", DT_R64F, contig, bcast_zero, contig, CUTENSOR_OP_ADD, false);
    g3_cell("flip", DT_R64F, neg, contig, contig, CUTENSOR_OP_ADD, false);
    g3_cell("out.T", DT_R64F, contig, contig, out_trans, CUTENSOR_OP_ADD, false);
    g3_cell("out-window", DT_R64F, contig, contig, out_window, CUTENSOR_OP_ADD, false);
    g3_cell("out-step", DT_R64F, contig, contig, out_step, CUTENSOR_OP_ADD, false);
    g3_cell("unary.T", DT_R64F, trans, trans, contig, CUTENSOR_OP_EXP, true);
    g3_cell("unary-window", DT_R64F, window, window, contig, CUTENSOR_OP_EXP, true);
    g3_cell("unary-step", DT_R64F, step, step, contig, CUTENSOR_OP_EXP, true);
    g3_cell("unary-out.T", DT_R64F, contig, contig, out_trans, CUTENSOR_OP_EXP, true);

    // rank-6: two transposed pairs
    {
        const int64_t d0 = 4, d1 = 5, d2 = 6;
        LSpec a{{d0, d1, d2}, {d1 * d2, 1, d1}, 0};  // pairs (1,2) swapped in memory
        LSpec b{{d0, d1, d2}, {d2, d0 * d2, 1}, 0};  // standard-ish
        g3_cell("rank6", DT_R64F, a, b, LSpec{{d0, d1, d2}, {d1 * d2, d2, 1}, 0}, CUTENSOR_OP_ADD, false);
    }

    // zero extent
    {
        LSpec z{{0, C}, {C, 1}, 0};
        g3_cell("zero-extent", DT_R64F, z, contig, z, CUTENSOR_OP_ADD, false);
    }
}

// ------------------------------------------------------------------------------------------- g4
static void g4_compose() {
    const int64_t R = 37, C = 53, n = R * C;
    LSpec contig{{R, C}, {C, 1}, 0};
    TensorSpec S = spec_of(DT_R64F, contig);
    auto av = make_vals((size_t)n, 51, false);
    auto bv = make_vals((size_t)n, 52, false);
    auto cv = make_vals((size_t)n, 53, false);
    TV ta{S, av}, tb{S, bv}, tc{S, cv};
    std::vector<uint8_t> hba = stage(ta), hbb = stage(tb), hbc = stage(tc);
    DBuf da(hba.size()), db(hbb.size()), dc(hbc.size()), dout((size_t)n * 8);
    da.upload(hba.data(), hba.size());
    db.upload(hbb.data(), hbb.size());
    dc.upload(hbc.data(), hbc.size());
    cudaMemset(dout.p, 0xFF, dout.bytes);

    // 1. unary via binary with gamma = 0 and C = nullptr
    {
        Plan p = make_binary_plan(H, S, CUTENSOR_OP_EXP, S, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD,
                                  CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 0.0, 0.0);
            cutensorStatus_t e = exec_binary(H, p, s, da.p, nullptr, dout.p);
            cudaError_t ce = cudaDeviceSynchronize();
            emit("G4", "unary gamma=0 C=nullptr", stage_str(p),
                 "exec=" + ab(e) + " cuda=" + cudaGetErrorString(ce));
        } else {
            emit("G4", "unary gamma=0 C=nullptr", stage_str(p), "");
        }
    }
    // 2. binary via trinary with gamma = 0 and C = nullptr
    {
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
            cutensorStatus_t e = exec_trinary(H, p, s, da.p, db.p, nullptr, dout.p);
            cudaError_t ce = cudaDeviceSynchronize();
            emit("G4", "binary gamma=0 C=nullptr", stage_str(p), "exec=" + ab(e) + " cuda=" + cudaGetErrorString(ce));
        } else {
            emit("G4", "binary gamma=0 C=nullptr", stage_str(p), "");
        }
    }
    // 3. in-place: A aliases D (expect INVALID_VALUE or garbage)
    {
        std::vector<uint8_t> tmp = hba;
        DBuf dinplace(tmp.size());
        dinplace.upload(tmp.data(), tmp.size());
        Plan p = make_binary_plan(H, S, CUTENSOR_OP_NEG, S, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD,
                                  CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 0.0, 0.0);
            cutensorStatus_t e = exec_binary(H, p, s, dinplace.p, dinplace.p, dinplace.p);
            cudaError_t ce = cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dinplace.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) bad += !close(out[(size_t)i], -av[(size_t)i], 1e-12, 1e-12);
            char b[128];
            snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
                     bad ? "FAIL" : "OK", bad);
            emit("G4", "in-place A=D neg", stage_str(p), b);
        } else {
            emit("G4", "in-place A=D neg", stage_str(p), "");
        }
    }
    // 4. fused c += a*b, C = D = c
    {
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_MUL, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 1.0);
            cutensorStatus_t e = exec_trinary(H, p, s, da.p, db.p, dc.p, dc.p);
            cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dc.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) bad += !close(out[(size_t)i], av[(size_t)i] * bv[(size_t)i] + cv[(size_t)i], 1e-12, 1e-12);
            emit("G4", "fused c += a*b", stage_str(p),
                 "exec=" + ab(e) + std::string(" verify=") + (bad ? "FAIL" : "OK") + "(" + std::to_string(bad) + ")");
        } else {
            emit("G4", "fused c += a*b", stage_str(p), "");
        }
    }
    // 4b. out-of-place D = a*b + c with separate C and D buffers
    {
        DBuf dsep((size_t)n * 8);
        dc.upload(hbc.data(), hbc.size());
        dsep.upload(hbc.data(), hbc.size());
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_MUL, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 1.0);
            cutensorStatus_t e = exec_trinary(H, p, s, da.p, db.p, dc.p, dsep.p);
            cudaError_t ce = cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dsep.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i)
                bad += !close(out[(size_t)i], av[(size_t)i] * bv[(size_t)i] + cv[(size_t)i], 1e-12, 1e-12);
            char b[128];
            snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
                     bad ? "FAIL" : "OK", bad);
            emit("G4", "out-of-place D = a*b + c (C!=D)", stage_str(p), b);
        } else {
            emit("G4", "out-of-place D = a*b + c (C!=D)", stage_str(p), "");
        }
    }
    // 5. scaled: D = 2A + 3C (C = D), alpha/gamma
    {
        DBuf dout2((size_t)n * 8);
        std::vector<uint8_t> hbc2 = hbc;
        dout2.upload(hbc2.data(), hbc2.size());
        Plan p = make_binary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD,
                                  CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 2.0, 0.0, 3.0);
            cutensorStatus_t e = exec_binary(H, p, s, da.p, dout2.p, dout2.p);
            cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dout2.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) bad += !close(out[(size_t)i], 2.0 * av[(size_t)i] + 3.0 * cv[(size_t)i], 1e-12, 1e-12);
            emit("G4", "D = 2A + 3C (C=D)", stage_str(p),
                 "exec=" + ab(e) + std::string(" verify=") + (bad ? "FAIL" : "OK") + "(" + std::to_string(bad) + ")");
        } else {
            emit("G4", "D = 2A + 3C (C=D)", stage_str(p), "");
        }
    }
    // 6. rank-0 second operand (scalar tensor): out = A + scalar
    {
        TensorSpec S0{CUDA_R_64F, {}, {}, 0};
        auto sv = make_vals(1, 54, false);
        TV ts{S0, sv};
        std::vector<uint8_t> hbs = stage(ts);
        DBuf ds(hbs.size());
        ds.upload(hbs.data(), hbs.size());
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S0, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
            cutensorStatus_t e = exec_trinary(H, p, s, da.p, ds.p, nullptr, dout.p);
            cudaError_t ce = cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dout.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) bad += !close(out[(size_t)i], av[(size_t)i] + sv[0], 1e-12, 1e-12);
            char b[128];
            snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
                     bad ? "FAIL" : "OK", bad);
            emit("G4", "rank-0 B scalar add", stage_str(p), b);
        } else {
            emit("G4", "rank-0 B scalar add", stage_str(p), "");
        }
    }
    // 6b. rank-1 extent-1 second operand (a "scalar" without rank 0)
    {
        TensorSpec S1{CUDA_R_64F, {1}, {1}, 0};
        auto sv = make_vals(1, 55, false);
        TV ts{S1, sv};
        std::vector<uint8_t> hbs = stage(ts);
        DBuf ds(hbs.size());
        ds.upload(hbs.data(), hbs.size());
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S1, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
            cutensorStatus_t e = exec_trinary(H, p, s, da.p, ds.p, dout.p, dout.p);
            cudaError_t ce = cudaDeviceSynchronize();
            TV t{S, {}};
            auto out = fetch(t, dout.p);
            int bad = 0;
            for (int64_t i = 0; i < n; ++i) bad += !close(out[(size_t)i], av[(size_t)i] + sv[0], 1e-12, 1e-12);
            char b[128];
            snprintf(b, sizeof b, "exec=%s cuda=%s verify=%s(%d)", ab(e).c_str(), cudaGetErrorString(ce),
                     bad ? "FAIL" : "OK", bad);
            emit("G4", "rank-1 extent-1 B scalar add", stage_str(p), b);
        } else {
            emit("G4", "rank-1 extent-1 B scalar add", stage_str(p), "");
        }
    }
    // 7. scalar types per dtype
    {
        const DType* dts[] = {&DT_R16F, &DT_R16BF, &DT_R32F, &DT_R64F, &DT_C32F, &DT_C64F, &DT_R32I, &DT_R8U};
        std::string line;
        for (const DType* dt : dts) {
            TensorSpec s{dt->v, {4, 3}, {3, 1}, 0};
            Plan p = make_binary_plan(H, s, CUTENSOR_OP_IDENTITY, s, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD,
                                      default_compute(dt->v));
            line += std::string(dt->name) + ":" + dt_name(p.scalar) + (p.ok ? " " : "(op-fail) ");
        }
        emit("G4", "scalar type per storage dtype", line, "");
    }
}

// plan creation cost with and without JIT, and plan-cache reuse
static void g4_jit_cost() {
    const int64_t R = 2048, C = 2048;
    TensorSpec S{CUDA_R_64F, {R, C}, {C, 1}, 0};
    for (int jit = 0; jit <= 1; ++jit) {
        cutensorJitMode_t mode = jit ? CUTENSOR_JIT_MODE_DEFAULT : CUTENSOR_JIT_MODE_NONE;
        double t0 = now_ms();
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F, mode);
        double t1 = now_ms();
        char b[160];
        snprintf(b, sizeof b, "plan_ms=%.3f ws=%llu ok=%d", t1 - t0, (unsigned long long)p.ws, (int)p.ok);
        emit("G4", std::string("plan cost jit=") + (jit ? "DEFAULT" : "NONE"), stage_str(p), b);
        if (!p.ok) continue;
        // time execute
        DBuf a((size_t)R * C * 8), b2((size_t)R * C * 8), o((size_t)R * C * 8);
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        double tc0 = now_ms();
        cutensorStatus_t e0 = exec_trinary(H, p, s, a.p, b2.p, o.p, o.p);
        double tc1 = now_ms();
        cudaDeviceSynchronize();
        for (int i = 0; i < 5; ++i) exec_trinary(H, p, s, a.p, b2.p, o.p, o.p);
        cudaDeviceSynchronize();
        EventTimer et;
        et.start();
        for (int i = 0; i < 20; ++i) exec_trinary(H, p, s, a.p, b2.p, o.p, o.p);
        double ms = et.stop_ms();
        char b3[160];
        snprintf(b3, sizeof b3, "exec=%.4fms first_exec_ms=%.3f", ms / 20.0, tc1 - tc0);
        emit("G4", std::string("exec jit=") + (jit ? "DEFAULT" : "NONE"), b3, "");
        (void)e0;
    }
    // second plan for the identical problem: the internal plan cache should make this cheap
    {
        double t0 = now_ms();
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        double t1 = now_ms();
        char b[128];
        snprintf(b, sizeof b, "second plan_ms=%.3f ok=%d", t1 - t0, (int)p.ok);
        emit("G4", "plan cache (identical problem)", stage_str(p), b);
    }
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cudaFree(0);
    cutensorStatus_t s = cutensorCreate(&H);
    if (s != CUTENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "cutensorCreate failed: %s\n", st_name(s));
        return 1;
    }
    printf("# cuTENSOR %d.%d.%d\n", (int)(cutensorGetVersion() / 10000), (int)((cutensorGetVersion() / 100) % 100),
           (int)(cutensorGetVersion() % 100));
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) groups.push_back(argv[i]);
    if (groups.empty()) groups = {"g1", "g2", "g3", "g4"};
    for (auto& g : groups) {
        if (g == "g1") g1_dtypes();
        else if (g == "g2") g2_promotion();
        else if (g == "g2b") g2b_mix();
        else if (g == "g2c") g2c_matrix();
        else if (g == "g2d") g2d_trinary();
        else if (g == "g3") g3_views();
        else if (g == "g4") {
            g4_compose();
            g4_jit_cost();
        } else fprintf(stderr, "unknown group %s\n", g.c_str());
    }
    cutensorDestroy(H);
    return 0;
}
