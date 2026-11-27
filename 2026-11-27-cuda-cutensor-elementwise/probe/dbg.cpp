// dbg.cpp -- minimal sanity cases for descriptor/mode conventions.
#include "common.h"

static cutensorHandle_t H = nullptr;

static std::string stage_str(const Plan& p) {
    auto ab = [](cutensorStatus_t s) { return s == CUTENSOR_STATUS_SUCCESS ? std::string("OK") : std::string(st_name(s)); };
    return "create=" + ab(p.s_op) + " pref=" + ab(p.s_pref) + " est=" + ab(p.s_est) + " plan=" + ab(p.s_plan);
}

static std::vector<cd> decode_bytes(void* p, size_t n, cudaDataType_t t) {
    std::vector<uint8_t> buf(n * (size_t)dt_bytes(t));
    cudaMemcpy(buf.data(), p, buf.size(), cudaMemcpyDeviceToHost);
    return decode(buf.data(), n, t);
}


static void show(const char* tag, const std::vector<cd>& got, const std::vector<cd>& want) {
    printf("%-24s", tag);
    for (size_t i = 0; i < std::min<size_t>(6, got.size()); ++i)
        printf(" got[%zu]=%.4f want=%.4f", i, got[i].real(), want[i].real());
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += !close(got[i], want[i], 1e-12, 1e-12);
    printf("  bad=%d/%zu\n", bad, got.size());
}

int main() {
    cudaFree(0);
    cutensorCreate(&H);

    // 1-D contiguous add
    {
        int64_t n = 10;
        TensorSpec S{CUDA_R_64F, {n}, {1}, 0};
        auto av = make_vals(n, 1, false), bv = make_vals(n, 2, false);
        std::vector<uint8_t> hba = encode(av, S.dt), hbb = encode(bv, S.dt);
        DBuf da(hba.size()), db(hbb.size()), dc(n * 8);
        da.upload(hba.data(), hba.size());
        db.upload(hbb.data(), hbb.size());
        cudaMemset(dc.p, 0xFF, dc.bytes);
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        printf("1d plan: %s\n", stage_str(p).c_str());
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        auto e = exec_trinary(H, p, s, da.p, db.p, dc.p, dc.p);
        printf("1d exec: %s\n", st_name(e));
        std::vector<cd> want(n);
        for (int64_t i = 0; i < n; ++i) want[i] = av[i] + bv[i];
        show("1d", decode_bytes(dc.p, n, S.dt), want);
    }

    // 2-D row-major [3,4]
    {
        int64_t R = 3, C = 4, n = R * C;
        TensorSpec S{CUDA_R_64F, {R, C}, {C, 1}, 0};
        auto av = make_vals(n, 3, false), bv = make_vals(n, 4, false);
        std::vector<uint8_t> hba = encode(av, S.dt), hbb = encode(bv, S.dt);
        DBuf da(hba.size()), db(hbb.size()), dc(n * 8);
        da.upload(hba.data(), hba.size());
        db.upload(hbb.data(), hbb.size());
        cudaMemset(dc.p, 0xFF, dc.bytes);
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        printf("2d row plan: %s\n", stage_str(p).c_str());
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        exec_trinary(H, p, s, da.p, db.p, dc.p, dc.p);
        std::vector<cd> want(n);
        for (int64_t i = 0; i < n; ++i) want[i] = av[i] + bv[i];
        show("2d row", decode_bytes(dc.p, n, S.dt), want);
    }

    // 2-D column-major layout (axis 0 fastest) with identity modes
    {
        int64_t R = 3, C = 4, n = R * C;
        TensorSpec S{CUDA_R_64F, {R, C}, {1, R}, 0};
        auto av = make_vals(n, 5, false), bv = make_vals(n, 6, false);
        // logical (r,c) lives at r + c*R
        std::vector<cd> am(n), bm(n);
        for (int64_t r = 0; r < R; ++r)
            for (int64_t c = 0; c < C; ++c) {
                am[r * C + c] = av[r + c * R];
                bm[r * C + c] = bv[r + c * R];
            }
        std::vector<uint8_t> hba = encode(am, S.dt), hbb = encode(bm, S.dt);
        DBuf da(hba.size()), db(hbb.size()), dc(n * 8);
        da.upload(hba.data(), hba.size());
        db.upload(hbb.data(), hbb.size());
        cudaMemset(dc.p, 0xFF, dc.bytes);
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        printf("2d col plan: %s\n", stage_str(p).c_str());
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        exec_trinary(H, p, s, da.p, db.p, dc.p, dc.p);
        std::vector<cd> got = decode_bytes(dc.p, n, S.dt);
        std::vector<cd> want(n);
        for (int64_t r = 0; r < R; ++r)
            for (int64_t c = 0; c < C; ++c) want[r * C + c] = av[r + c * R] + bv[r + c * R];
        show("2d col (identity modes)", got, want);
    }

    // 2-D row-major [37,53]
    {
        int64_t R = 37, C = 53, n = R * C;
        TensorSpec S{CUDA_R_64F, {R, C}, {C, 1}, 0};
        auto av = make_vals(n, 7, false), bv = make_vals(n, 8, false);
        std::vector<uint8_t> hba = encode(av, S.dt), hbb = encode(bv, S.dt);
        DBuf da(hba.size()), db(hbb.size()), dc(n * 8);
        da.upload(hba.data(), hba.size());
        db.upload(hbb.data(), hbb.size());
        cudaMemset(dc.p, 0xFF, dc.bytes);
        Plan p = make_trinary_plan(H, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY, S, CUTENSOR_OP_IDENTITY,
                                   CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        printf("37x53 plan: %s\n", stage_str(p).c_str());
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        auto e = exec_trinary(H, p, s, da.p, db.p, dc.p, dc.p);
        printf("37x53 exec: %s\n", st_name(e));
        std::vector<cd> want(n);
        for (int64_t i = 0; i < n; ++i) want[i] = av[i] + bv[i];
        auto got = decode_bytes(dc.p, n, S.dt);
        int bad = 0;
        for (int64_t i = 0; i < n; ++i) bad += !close(got[i], want[i], 1e-12, 1e-12);
        printf("37x53 bad=%d/%lld  got[0..3]=%.4f %.4f %.4f %.4f want=%.4f %.4f %.4f %.4f\n", bad, (long long)n, got[0].real(),
               got[1].real(), got[2].real(), got[3].real(), want[0].real(), want[1].real(), want[2].real(), want[3].real());
    }


    // exact g3-style path with stage()/fetch()
    {
        const int64_t R = 37, C = 53;
        TensorSpec sa{CUDA_R_64F, {R, C}, {C, 1}, 0};
        TensorSpec sb{CUDA_R_64F, {R, C}, {C, 1}, 0};
        TensorSpec so{CUDA_R_64F, {R, C}, {C, 1}, 0};
        TV ta{sa, {}}, tb{sb, {}}, tout{so, {}};
        ta.logical = make_vals((size_t)(R * C), 41, false, false);
        tb.logical = make_vals((size_t)(R * C), 42, false, false);
        std::vector<uint8_t> hba = stage(ta), hbb = stage(tb);
        printf("stage sizes %zu %zu bytes=%d\n", hba.size(), hbb.size(), dt_bytes(ta.spec.dt));
        DBuf da(hba.size()), db(hbb.size()), dout((size_t)(R * C) * 8);
        da.upload(hba.data(), hba.size());
        db.upload(hbb.data(), hbb.size());
        cudaMemset(dout.p, 0xFF, dout.bytes);
        Plan p = make_trinary_plan(H, ta.spec, CUTENSOR_OP_IDENTITY, tb.spec, CUTENSOR_OP_IDENTITY, tout.spec,
                                   CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F);
        printf("g3style plan: %s\n", stage_str(p).c_str());
        Scalars s;
        fill_scalars(s, p.scalar, 1.0, 1.0, 0.0);
        auto e = exec_trinary(H, p, s, da.p, db.p, dout.p, dout.p);
        printf("g3style exec: %s\n", st_name(e));
        auto out = fetch(tout, dout.p);
        int bad = 0;
        for (int64_t i = 0; i < R * C; ++i)
            bad += !close(out[(size_t)i], ta.logical[(size_t)i] + tb.logical[(size_t)i], 1e-12, 1e-12);
        printf("g3style bad=%d  out[1]=%.6f want=%.6f\n", bad, out[1].real(), ta.logical[1].real() + tb.logical[1].real());
    }

    cutensorDestroy(H);
    return 0;
}
