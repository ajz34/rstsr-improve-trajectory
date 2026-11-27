// bench.cpp -- cuTENSOR elementwise timing against the same cells as the rstsr hand-kernel bench.
//
// One CSV row per cell: plan cost, first-exec, steady-state per-exec (CUDA events, median of reps),
// host enqueue cost per call, achieved GB/s, and a sampled correctness check.
#include "common.h"

static cutensorHandle_t H = nullptr;

struct Sz {
    const char* name;
    int64_t R, C;
};

// sizes: S small, M 2048^2, L 4096^2, O* odd
static const Sz SIZES[] = {{"S", 100, 100},   {"M", 2048, 2048}, {"L", 4096, 4096},
                           {"O1", 1000, 1001}, {"O2", 2047, 2053}, {"O3", 4093, 4099}};
static const char* SZ_NAMES[] = {"S", "M", "L", "O1", "O2", "O3"};

// layouts
enum { L_CONTIG = 0, L_AT, L_BT, L_BOTH_T, L_WIN, L_STEP, L_BCAST, L_OUT_T, L_N };
static const char* LAYOUT_NAMES[] = {"contig", "At", "Bt", "bothT", "win", "step", "bcast", "outT"};

struct LSet {
    TensorSpec A, B, CD;
};

static LSet build_layout(int layout, int64_t R, int64_t C, const DType& dt) {
    LSet s;
    s.A = TensorSpec{dt.v, {R, C}, {C, 1}, 0};
    s.B = TensorSpec{dt.v, {R, C}, {C, 1}, 0};
    s.CD = TensorSpec{dt.v, {R, C}, {C, 1}, 0};
    switch (layout) {
        case L_AT: s.A = TensorSpec{dt.v, {R, C}, {1, C}, 0}; break;
        case L_BT: s.B = TensorSpec{dt.v, {R, C}, {1, C}, 0}; break;
        case L_BOTH_T:
            s.A = TensorSpec{dt.v, {R, C}, {1, C}, 0};
            s.B = TensorSpec{dt.v, {R, C}, {1, C}, 0};
            break;
        case L_WIN:
            s.A = TensorSpec{dt.v, {R, C}, {C + 20, 1}, 10 * (C + 20) + 10};
            s.B = TensorSpec{dt.v, {R, C}, {C + 20, 1}, 10 * (C + 20) + 10};
            break;
        case L_STEP:
            s.A = TensorSpec{dt.v, {R, C}, {4 * C, 2}, 0};
            s.B = TensorSpec{dt.v, {R, C}, {4 * C, 2}, 0};
            break;
        case L_BCAST: s.B = TensorSpec{dt.v, {R, 1}, {1, 1}, 0}; break;
        case L_OUT_T: s.CD = TensorSpec{dt.v, {R, C}, {1, R}, 0}; break;
        default: break;
    }
    return s;
}

struct BCell {
    std::string name;
    int kind = 1;  // 0 unary(bin api), 1 binary(trinary), 2 trinary inplace, 3 trinary oop, 4 bin api w/ C
    const DType* dt = &DT_R64F;
    cutensorComputeDescriptor_t comp = CUTENSOR_COMPUTE_DESC_64F;
    cutensorOperator_t op = CUTENSOR_OP_ADD;
    int layout = L_CONTIG;
    int size_id = 1;
    cutensorJitMode_t jit = CUTENSOR_JIT_MODE_NONE;
    bool pos = false;
};

static inline int64_t off_for(const TensorSpec& s, int64_t r, int64_t c) {
    int64_t rr = (!s.ext.empty() && s.ext[0] == 1) ? 0 : r;
    int64_t cc = (s.ext.size() > 1 && s.ext[1] == 1) ? 0 : c;
    return s.off + rr * s.str[0] + (s.ext.size() > 1 ? cc * s.str[1] : 0);
}

static int accesses(int kind) {
    switch (kind) {
        case 0: return 2;  // read A, write D
        case 1: return 3;  // read A, read B, write D
        case 2: return 4;  // read A, read B, read+write C(=D)
        case 3: return 4;  // read A, read B, read C, write D
        default: return 3;  // read A, read B(=C), write D
    }
}

struct Res {
    bool ok = false;
    double plan_ms = 0, first_ms = 0, med_ms = 0, host_us = 0, gbps = 0;
    int bad = -1;
    std::string status;
};

static Res run_cell(const BCell& c, bool verify) {
    Res res;
    const Sz& sz = SIZES[c.size_id];
    int64_t R = sz.R, C = sz.C, n = R * C;
    LSet ls = build_layout(c.layout, R, C, *c.dt);
    size_t an = (size_t)ls.A.extent_elems(), bn = (size_t)ls.B.extent_elems(), cn = (size_t)ls.CD.extent_elems();
    size_t ab = an * (size_t)c.dt->bytes, bb = bn * (size_t)c.dt->bytes, cb = cn * (size_t)c.dt->bytes;

    DBuf da(ab), db(bb), dd(cb), dc(c.kind == 3 ? cb : 256), dscratch(c.kind >= 3 ? cb : 256);
    auto ha = make_vals(an, 101 + c.size_id, c.dt->is_complex, c.pos);
    auto hb = make_vals(bn, 202 + c.size_id, c.dt->is_complex, c.pos);
    auto hc = make_vals(cn, 303 + c.size_id, c.dt->is_complex);
    std::vector<uint8_t> ba = encode(ha, c.dt->v), bbb = encode(hb, c.dt->v), bc = encode(hc, c.dt->v);
    da.upload(ba.data(), ba.size());
    db.upload(bbb.data(), bbb.size());
    dd.upload(bc.data(), bc.size());
    int eb = c.dt->bytes;
    void* Aptr = (char*)da.p + ls.A.off * eb;
    void* Bptr = (char*)db.p + ls.B.off * eb;
    void* Cptr = (char*)dd.p + ls.CD.off * eb;
    void* Dptr = Cptr;
    if (c.kind == 3) {
        dscratch.upload(bc.data(), bc.size());
        Cptr = (char*)dscratch.p + ls.CD.off * eb;
        Dptr = (char*)dd.p + ls.CD.off * eb;
    } else if (c.kind == 4) {
        Cptr = Bptr;  // C is the second operand
        Dptr = (char*)dd.p + ls.CD.off * eb;
    }

    Plan p;
    double t0 = now_ms();
    if (c.kind == 0) {
        p = make_binary_plan(H, ls.A, c.op, ls.CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, c.comp, c.jit);
    } else if (c.kind == 4) {
        p = make_binary_plan(H, ls.A, CUTENSOR_OP_IDENTITY, ls.CD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, c.comp,
                             c.jit);
    } else {
        p = make_trinary_plan(H, ls.A, CUTENSOR_OP_IDENTITY, ls.B, CUTENSOR_OP_IDENTITY, ls.CD,
                              CUTENSOR_OP_IDENTITY, (c.kind == 0 ? CUTENSOR_OP_ADD : c.op), CUTENSOR_OP_ADD, c.comp,
                              c.jit);
    }
    res.plan_ms = now_ms() - t0;
    if (!p.ok) {
        res.status = plan_status(p);
        return res;
    }

    Scalars s;
    fill_scalars(s, p.scalar, 1.0, 1.0, (c.kind == 2 || c.kind == 3 || c.kind == 4) ? 1.0 : 0.0);
    auto exec = [&]() {
        if (c.kind == 0 || c.kind == 4) return exec_binary(H, p, s, Aptr, Cptr, Dptr);
        return exec_trinary(H, p, s, Aptr, Bptr, Cptr, Dptr);
    };

    double f0 = now_ms();
    cutensorStatus_t e0 = exec();
    cudaError_t ce = cudaDeviceSynchronize();
    res.first_ms = now_ms() - f0;
    if (e0 != CUTENSOR_STATUS_SUCCESS || ce != cudaSuccess) return res;

    if (verify) {
        std::vector<uint8_t> outb(cb);
        cudaMemcpy(outb.data(), dd.p, cb, cudaMemcpyDeviceToHost);
        std::vector<cd> ho = decode(outb.data(), cn, c.dt->v);
        std::mt19937_64 rng(7);
        int bad = 0;
        for (int k = 0; k < 512; ++k) {
            int64_t r = (int64_t)(rng() % (uint64_t)R), col = (int64_t)(rng() % (uint64_t)C);
            cd a = decode(encode({ha[(size_t)off_for(ls.A, r, col)]}, c.dt->v).data(), 1, c.dt->v)[0];
            cd bb2 = decode(encode({hb[(size_t)off_for(ls.B, r, col)]}, c.dt->v).data(), 1, c.dt->v)[0];
            cd cc = decode(encode({hc[(size_t)off_for(ls.CD, r, col)]}, c.dt->v).data(), 1, c.dt->v)[0];
            cd want;
            if (c.kind == 0) want = ref_unary(c.op, a);
            else if (c.kind == 1) want = ref_binary(c.op, a, bb2);
            else if (c.kind == 2 || c.kind == 3) want = ref_binary(CUTENSOR_OP_ADD, ref_binary(CUTENSOR_OP_MUL, a, bb2), cc);
            else want = ref_binary(CUTENSOR_OP_ADD, a, bb2);
            cd stored = decode(encode({want}, c.dt->v).data(), 1, c.dt->v)[0];
            cd got = ho[(size_t)off_for(ls.CD, r, col)];
            if (!close(got, stored, tol_for(c.dt->v), tol_for(c.dt->v))) ++bad;
        }
        res.bad = bad;
    }
    res.ok = true;

    for (int i = 0; i < 10; ++i) exec();
    cudaDeviceSynchronize();

    std::vector<double> per;
    const int it = 50, reps = 7;
    for (int rep = 0; rep < reps; ++rep) {
        EventTimer et;
        et.start();
        for (int i = 0; i < it; ++i) exec();
        per.push_back(et.stop_ms() / it);
    }
    std::sort(per.begin(), per.end());
    res.med_ms = per[per.size() / 2];

    cudaDeviceSynchronize();
    double h0 = now_ms();
    for (int i = 0; i < 200; ++i) exec();
    double h1 = now_ms();
    cudaDeviceSynchronize();
    res.host_us = (h1 - h0) * 1000.0 / 200.0;

    res.gbps = (double)accesses(c.kind) * (double)n * (double)c.dt->bytes / (res.med_ms * 1e6);
    return res;
}

int main(int argc, char** argv) {
    cudaFree(0);
    if (cutensorCreate(&H) != CUTENSOR_STATUS_SUCCESS) return 1;
    bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    FILE* csv = fopen("results/cutensor_bench.csv", "w");
    fprintf(csv, "cell,kind,dtype,layout,size,R,C,n,plan_ms,first_ms,med_ms,host_us,gbps,bad\n");

    std::vector<BCell> cells;
    auto add = [&](const char* opname, int kind, const DType& dt, cutensorComputeDescriptor_t comp,
                   cutensorOperator_t op, int layout, int sid, cutensorJitMode_t jit = CUTENSOR_JIT_MODE_NONE) {
        BCell c;
        char nm[128];
        snprintf(nm, sizeof nm, "%s_%s_%s_%s%s%s", opname, dt.name, LAYOUT_NAMES[layout], SZ_NAMES[sid],
                 kind == 4 ? "_binapi" : "", jit == CUTENSOR_JIT_MODE_NONE ? "" : "_jit");
        c.name = nm;
        c.kind = kind;
        c.dt = &dt;
        c.comp = comp;
        c.op = op;
        c.layout = layout;
        c.size_id = sid;
        c.jit = jit;
        cells.push_back(c);
    };

    // M: operator/dtype matrix
    add("copy", 0, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_IDENTITY, L_CONTIG, 1);
    add("exp", 0, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_EXP, L_CONTIG, 1);
    add("sqrt", 0, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_SQRT, L_CONTIG, 1);
    cells.back().pos = true;
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("add", 4, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("mul", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MUL, L_CONTIG, 1);
    add("max", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MAX, L_CONTIG, 1);
    add("min", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MIN, L_CONTIG, 1);
    add("muladd", 2, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MUL, L_CONTIG, 1);
    add("muladdoop", 3, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MUL, L_CONTIG, 1);
    add("add", 1, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("exp", 0, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_EXP, L_CONTIG, 1);
    add("muladd", 2, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_MUL, L_CONTIG, 1);
    add("add", 1, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("add", 1, DT_C32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("add", 1, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("add", 1, DT_R16BF, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 1);
    add("muladd", 2, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_MUL, L_CONTIG, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_AT, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_BT, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_BOTH_T, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_WIN, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_STEP, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_BCAST, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_OUT_T, 1);
    add("add", 1, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_AT, 1);
    add("add", 1, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_STEP, 1);
    add("add", 1, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_AT, 1);
    add("add", 1, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_AT, 1);
    add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, 1, CUTENSOR_JIT_MODE_DEFAULT);

    // odd sizes and large size, contiguous f64
    for (int sid : {0, 3, 4, 5, 2}) {
        add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, sid);
        add("copy", 0, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_IDENTITY, L_CONTIG, sid);
    }
    add("add", 1, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 5);
    add("add", 1, DT_R32F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 2);
    add("muladdoop", 3, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_MUL, L_CONTIG, 2);
    // small-size layout matrix
    for (int l : {L_AT, L_WIN, L_STEP, L_BCAST, L_OUT_T}) add("add", 1, DT_R64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, l, 0);
    add("add", 1, DT_R16F, CUTENSOR_COMPUTE_DESC_32F, CUTENSOR_OP_ADD, L_CONTIG, 0);
    add("add", 1, DT_C64F, CUTENSOR_COMPUTE_DESC_64F, CUTENSOR_OP_ADD, L_CONTIG, 0);

    // 5 s power warmup on f64 add at M
    {
        BCell w;
        w.name = "warmup";
        w.kind = 1;
        w.dt = &DT_R64F;
        w.comp = CUTENSOR_COMPUTE_DESC_64F;
        w.op = CUTENSOR_OP_ADD;
        w.size_id = 1;
        LSet ls = build_layout(L_CONTIG, 2048, 2048, DT_R64F);
        size_t nb = 2048 * 2048 * 8;
        DBuf a(nb), b(nb), d(nb);
        Plan p = make_trinary_plan(H, ls.A, CUTENSOR_OP_IDENTITY, ls.B, CUTENSOR_OP_IDENTITY, ls.CD,
                                   CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        if (p.ok) {
            Scalars s;
            fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
            double t0 = now_ms();
            while (now_ms() - t0 < 5000) {
                for (int i = 0; i < 20; ++i) exec_trinary(H, p, s, a.p, b.p, d.p, d.p);
                cudaDeviceSynchronize();
            }
        }
    }

    int failed = 0;
    for (auto& c : cells) {
        Res r = run_cell(c, true);
        const Sz& sz = SIZES[c.size_id];
        if (!r.ok) {
            fprintf(csv, "%s,%d,%s,%s,%s,%lld,%lld,%lld,%.4f,,,,,\n", c.name.c_str(), c.kind, c.dt->name,
                    LAYOUT_NAMES[c.layout], SZ_NAMES[c.size_id], (long long)sz.R, (long long)sz.C,
                    (long long)(sz.R * sz.C), r.plan_ms);
            printf("%-28s FAIL %s\n", c.name.c_str(), r.status.c_str());
            ++failed;
            continue;
        }
        fprintf(csv, "%s,%d,%s,%s,%s,%lld,%lld,%lld,%.4f,%.4f,%.5f,%.2f,%.1f,%d\n", c.name.c_str(), c.kind,
                c.dt->name, LAYOUT_NAMES[c.layout], SZ_NAMES[c.size_id], (long long)sz.R, (long long)sz.C,
                (long long)(sz.R * sz.C), r.plan_ms, r.first_ms, r.med_ms, r.host_us, r.gbps, r.bad);
        printf("%-28s plan=%7.3fms first=%7.3fms med=%8.4fms host=%6.2fus %7.1f GB/s bad=%d\n", c.name.c_str(),
               r.plan_ms, r.first_ms, r.med_ms, r.host_us, r.gbps, r.bad);
        fflush(stdout);
    }
    fclose(csv);
    cutensorDestroy(H);
    printf("# %d cells, %d failed\n", (int)cells.size(), failed);
    return 0;
}
