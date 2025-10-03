// main.cpp
#include "depth_location_code.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstring>
#include <array>
#include <atomic>
#include <type_traits>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

using Clock = std::chrono::steady_clock;

// prevent reordering/hoisting across timed regions
static inline void compiler_fence() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Maximum legal depth for a given code width: 3*depth payload + 1 delimiter <= bit-width
template <typename Code>
static constexpr uint8_t max_depth_for_code() {
    return static_cast<uint8_t>((sizeof(Code) * 8 - 1) / 3);
}

enum class Variant { LOOP, MAGIC, LUT, BMI2, OPT, OPT2, OPT3, AUTO };

static constexpr const char* kVariantNames[] = {
    "LOOP","MAGIC","LUT","BMI2","OPT","OPT2","OPT3","AUTO"
};

static Variant parse_variant(const char* label) {
    if (std::strcmp(label, "LOOP")  == 0) return Variant::LOOP;
    if (std::strcmp(label, "MAGIC") == 0) return Variant::MAGIC;
    if (std::strcmp(label, "LUT")   == 0) return Variant::LUT;
    if (std::strcmp(label, "BMI2")  == 0) return Variant::BMI2;
    if (std::strcmp(label, "OPT")   == 0) return Variant::OPT;
    if (std::strcmp(label, "OPT2")  == 0) return Variant::OPT2;
    if (std::strcmp(label, "OPT3")  == 0) return Variant::OPT3;
    return Variant::AUTO;
}

template <typename DLC>
struct BoundFns {
    using Code = typename DLC::code_type_alias;
    using Pos  = typename DLC::pos_type_alias;
    using Enc  = Code (DLC::*)(Pos,Pos,Pos,uint8_t);
    using Dec  = std::tuple<Pos,Pos,Pos> (DLC::*)(Code);
    Enc enc;
    Dec dec;
};

template <typename DLC>
static BoundFns<DLC> bind_variant(Variant v) {
    using F = BoundFns<DLC>;
    switch (v) {
        case Variant::LOOP:  return F{ &DLC::encode_loop,       &DLC::decode_loop       };
        case Variant::MAGIC: return F{ &DLC::encode_magic,      &DLC::decode_magic      };
        case Variant::LUT:   return F{ &DLC::encode_LUT,        &DLC::decode_LUT        };
        case Variant::BMI2:  return F{ &DLC::encode_BMI2,       &DLC::decode_BMI2       };
        case Variant::OPT:   return F{ &DLC::encode_magic_opt,  &DLC::decode_magic      };
        case Variant::OPT2:  return F{ &DLC::encode_magic_opt2, &DLC::decode_magic_opt2 };
        case Variant::OPT3:  return F{ &DLC::encode_magic_opt3, &DLC::decode_magic_opt3 };
        case Variant::AUTO:
            if constexpr (sizeof(typename DLC::code_type_alias) <= 4)
                return F{ &DLC::encode_LUT,        &DLC::decode_magic_opt2 };
            else
                return F{ &DLC::encode_magic_opt2, &DLC::decode_magic_opt2 };
    }
    return F{ &DLC::encode_magic, &DLC::decode_magic };
}

// ---------- Accuracy pass ----------
template <typename DLC>
static void accuracy_pass(const char* code_label, std::size_t N) {
    using Code = typename DLC::code_type_alias;
    using Pos  = typename DLC::pos_type_alias;

    constexpr uint8_t kDepthCap   = max_depth_for_code<Code>();
    constexpr uint8_t kGlobalMax  = 21;
    const uint8_t depth_hi        = (kDepthCap < kGlobalMax) ? kDepthCap : kGlobalMax;

    std::mt19937_64 rng(0xA11ACCu ^ (sizeof(Code)<<8));
    std::uniform_int_distribution<int> depth_dist(1, depth_hi);

    DLC dlc;

    std::array<Variant,8> variants = {
        Variant::LOOP, Variant::MAGIC, Variant::LUT, Variant::BMI2,
        Variant::OPT, Variant::OPT2, Variant::OPT3, Variant::AUTO
    };
    std::array<BoundFns<DLC>,8> fns;
    for (std::size_t i=0;i<variants.size();++i) fns[i] = bind_variant<DLC>(variants[i]);

    std::array<std::size_t,8> mask_err{};      // decoded vs masked input mismatch
    std::size_t key_disagree_err = 0;          // any key != MAGIC key

    std::vector<Code> kk(variants.size());
    std::vector<std::tuple<Pos,Pos,Pos>> dd(variants.size());

    for (std::size_t i=0;i<N;++i) {
        const uint8_t d = static_cast<uint8_t>(depth_dist(rng));
        const uint32_t maxCoord = (d==0)?0u:((1u<<d)-1u);
        std::uniform_int_distribution<uint32_t> coord(0u, maxCoord);
        const Pos x = static_cast<Pos>(coord(rng));
        const Pos y = static_cast<Pos>(coord(rng));
        const Pos z = static_cast<Pos>(coord(rng));
        const Pos mask = static_cast<Pos>(maxCoord);

        for (std::size_t v=0; v<variants.size(); ++v)
            kk[v] = (dlc.*(fns[v].enc))(x,y,z,d);

        const Code kref = kk[1]; // MAGIC index=1
        for (std::size_t v=0; v<variants.size(); ++v) {
            if (kk[v] != kref) { ++key_disagree_err; break; }
        }

        for (std::size_t v=0; v<variants.size(); ++v) {
            dd[v] = (dlc.*(fns[v].dec))(kk[v]);
            const Pos dx = std::get<0>(dd[v]);
            const Pos dy = std::get<1>(dd[v]);
            const Pos dz = std::get<2>(dd[v]);
            if ((dx != (x & mask)) || (dy != (y & mask)) || (dz != (z & mask))) {
                ++mask_err[v];
            }
        }
    }

    std::printf("\n== ACCURACY CHECK for codeType=%s (N=%zu) ==\n", code_label, N);
    for (std::size_t v=0; v<variants.size(); ++v) {
        std::printf("  %-6s: decode_mismatch=%zu\n", kVariantNames[v], mask_err[v]);
    }
    std::printf("  key disagreement (any variant vs MAGIC): %zu\n", key_disagree_err);
}

// Measure callable until min_ms reached
template <class F>
static std::pair<double, std::size_t> time_until_ms(F&& fn, double min_ms) {
    std::size_t reps = 0;
    auto t0 = Clock::now();
    double elapsed_ms = 0.0;
    do {
        fn();
        ++reps;
        auto t1 = Clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } while (elapsed_ms < min_ms);
    return { elapsed_ms / 1000.0, reps };
}

// ---------- Timing ----------
template <typename DLC>
static void bench_variant(const char* label, std::size_t N, double min_ms = 50.0) {
    using Code = typename DLC::code_type_alias;
    using Pos  = typename DLC::pos_type_alias;

    constexpr uint8_t kDepthCap   = max_depth_for_code<Code>();
    constexpr uint8_t kGlobalMax  = 21;
    const uint8_t depth_hi        = (kDepthCap < kGlobalMax) ? kDepthCap : kGlobalMax;

    std::mt19937_64 rng(0xC0FFEEULL ^ (N * 1315423911ULL));
    std::uniform_int_distribution<int> depth_dist(1, depth_hi);

    std::vector<uint8_t> depths(N);
    std::vector<Pos>     xs(N), ys(N), zs(N);
    std::vector<Code>    keys(N);
    std::vector<Pos>     dx(N), dy(N), dz(N);

    for (std::size_t i = 0; i < N; ++i) {
        uint8_t d = static_cast<uint8_t>(depth_dist(rng));
        depths[i] = d;
        const uint32_t maxCoord = (d == 0) ? 0u : ((1u << d) - 1u);
        std::uniform_int_distribution<uint32_t> coord(0u, maxCoord);
        xs[i] = static_cast<Pos>(coord(rng));
        ys[i] = static_cast<Pos>(coord(rng));
        zs[i] = static_cast<Pos>(coord(rng));
    }

    DLC dlc;
    const Variant V = parse_variant(label);
    auto f = bind_variant<DLC>(V);

    // warmup
    {
        const std::size_t W = std::min<std::size_t>(N, 1000);
        for (std::size_t i = 0; i < W; ++i) {
            keys[i] = (dlc.*(f.enc))(xs[i], ys[i], zs[i], depths[i]);
        }
        for (std::size_t i = 0; i < W; ++i) {
            auto t  = (dlc.*(f.dec))(keys[i]);
            dx[i] = std::get<0>(t);
            dy[i] = std::get<1>(t);
            dz[i] = std::get<2>(t);
        }
    }

    auto encode_pass = [&](){
        for (std::size_t i = 0; i < N; ++i)
            keys[i] = (dlc.*(f.enc))(xs[i], ys[i], zs[i], depths[i]);
    };
    compiler_fence();
    auto [enc_secs, enc_reps] = time_until_ms(encode_pass, min_ms);
    compiler_fence();

    auto decode_pass = [&](){
        for (std::size_t i = 0; i < N; ++i) {
            auto tup = (dlc.*(f.dec))(keys[i]);
            dx[i] = std::get<0>(tup);
            dy[i] = std::get<1>(tup);
            dz[i] = std::get<2>(tup);
        }
    };
    compiler_fence();
    auto [dec_secs, dec_reps] = time_until_ms(decode_pass, min_ms);
    compiler_fence();

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(N, 10000); ++i) {
        const auto mask = (depths[i]==0)?0u:((1u<<depths[i])-1u);
        if ((dx[i] != (xs[i] & mask)) || (dy[i] != (ys[i] & mask)) || (dz[i] != (zs[i] & mask))) ++mismatches;
    }

    const double enc_ops = static_cast<double>(N) * static_cast<double>(enc_reps);
    const double dec_ops = static_cast<double>(N) * static_cast<double>(dec_reps);

    const double enc_ns_per_op = (enc_secs * 1e9) / enc_ops;
    const double dec_ns_per_op = (dec_secs * 1e9) / dec_ops;

    const double enc_mops = enc_ops / (enc_secs * 1e6);
    const double dec_mops = dec_ops / (dec_secs * 1e6);

    std::printf("  [%s][N=%llu, reps=%llu] "
                "encode: %7.2f ms | %7.2f ns/op | %7.2f Mops/s  ||  "
                "decode: %7.2f ms | %7.2f ns/op | %7.2f Mops/s | mismatches=%llu\n",
        label,
        (unsigned long long)N,
        (unsigned long long)enc_reps,
        enc_secs * 1e3, enc_ns_per_op, enc_mops,
        dec_secs * 1e3, dec_ns_per_op, dec_mops,
        (unsigned long long)mismatches);
}

// ---------- Generic batch benches that ALWAYS call dlc.encode_batch/decode_batch ----------
template <typename DLC>
static void bench_any_batch_encode(std::size_t N, uint8_t depth) {
    using Code = typename DLC::code_type_alias;
    using Pos  = typename DLC::pos_type_alias;

    std::mt19937_64 rng(0xE11AAF00DULL ^ (N*17));
    const uint32_t mask = (depth >= 21) ? ((1u<<21) - 1u) : ((1u << depth) - 1u);
    std::uniform_int_distribution<uint32_t> coord(0u, mask);

    std::vector<Pos> x(N), y(N), z(N);
    for (std::size_t i=0;i<N;++i) { x[i]=coord(rng); y[i]=coord(rng); z[i]=coord(rng); }

    std::vector<Code> out_scalar(N), out_batch(N);
    DLC dlc;

    // scalar "reference"
    auto scalar_pass = [&](){
        for (std::size_t i=0;i<N;++i)
            out_scalar[i] = dlc.encode_magic_opt2(x[i], y[i], z[i], depth);
    };
    compiler_fence();
    auto [secs_scalar, reps_scalar] = time_until_ms(scalar_pass, 50.0);
    compiler_fence();

    // true batch (uses SIMD for u64; and, with updated header, SIMD LUT path for <=32-bit)
    auto batch_pass = [&](){
        dlc.encode_batch(x.data(), y.data(), z.data(), depth, out_batch.data(), N);
    };
    compiler_fence();
    auto [secs_batch, reps_batch] = time_until_ms(batch_pass, 50.0);
    compiler_fence();

    // correctness
    std::size_t mismatches = 0;
    for (std::size_t i=0;i<N;++i) if (out_scalar[i] != out_batch[i]) ++mismatches;

    const double ops_scalar = double(N) * reps_scalar;
    const double ops_batch  = double(N) * reps_batch;

    std::printf("\n== Batch encode (Code=%s, depth=%u) ==\n",
                std::is_same<Code,uint64_t>::value ? "u64" :
                std::is_same<Code,uint32_t>::value ? "u32" :
                std::is_same<Code,uint16_t>::value ? "u16" : "u8",
                depth);
    std::printf("  Scalar encode_opt2:   %7.2f ms | %7.2f ns/op | %7.2f Mops/s\n",
                secs_scalar*1e3, (secs_scalar*1e9)/ops_scalar, ops_scalar/(secs_scalar*1e6));
    std::printf("  Batch  encode_batch:  %7.2f ms | %7.2f ns/op | %7.2f Mops/s\n",
                secs_batch*1e3, (secs_batch*1e9)/ops_batch, ops_batch/(secs_batch*1e6));
    std::printf("  correctness mismatches: %llu\n",
                (unsigned long long)mismatches);
}

template <typename DLC>
static void bench_any_batch_decode(std::size_t N, uint8_t depth) {
    using Code = typename DLC::code_type_alias;
    using Pos  = typename DLC::pos_type_alias;

    std::mt19937_64 rng(0xC0DEC0DEULL ^ (N*29));
    const uint32_t mask = (depth >= 21) ? ((1u<<21) - 1u) : ((1u << depth) - 1u);
    std::uniform_int_distribution<uint32_t> coord(0u, mask);

    std::vector<Pos> x_in(N), y_in(N), z_in(N);
    for (std::size_t i=0;i<N;++i) { x_in[i]=coord(rng); y_in[i]=coord(rng); z_in[i]=coord(rng); }

    std::vector<Code> keys(N);
    DLC dlc;

    // Make legal keys for the CodeT we are testing
    for (std::size_t i=0;i<N;++i)
        keys[i] = dlc.encode_magic_opt2(x_in[i], y_in[i], z_in[i], depth);

    std::vector<Pos> xs_scalar(N), ys_scalar(N), zs_scalar(N);
    std::vector<Pos> xs_batch(N),  ys_batch(N),  zs_batch(N);

    // scalar reference decode
    auto scalar_pass = [&](){
        for (std::size_t i=0;i<N;++i) {
            auto t = dlc.decode_magic_opt2(keys[i]);
            xs_scalar[i] = std::get<0>(t);
            ys_scalar[i] = std::get<1>(t);
            zs_scalar[i] = std::get<2>(t);
        }
    };
    compiler_fence();
    auto [secs_scalar, reps_scalar] = time_until_ms(scalar_pass, 50.0);
    compiler_fence();

    // true batch
    auto batch_pass = [&](){
        dlc.decode_batch(keys.data(), xs_batch.data(), ys_batch.data(), zs_batch.data(), N);
    };
    compiler_fence();
    auto [secs_batch, reps_batch] = time_until_ms(batch_pass, 50.0);
    compiler_fence();

    std::size_t mismatches = 0;
    for (std::size_t i=0;i<N;++i)
        if (xs_scalar[i]!=xs_batch[i] || ys_scalar[i]!=ys_batch[i] || zs_scalar[i]!=zs_batch[i]) ++mismatches;

    const double ops_scalar = double(N) * reps_scalar;
    const double ops_batch  = double(N) * reps_batch;

    std::printf("\n== Batch decode (Code=%s, depth=%u) ==\n",
                std::is_same<Code,uint64_t>::value ? "u64" :
                std::is_same<Code,uint32_t>::value ? "u32" :
                std::is_same<Code,uint16_t>::value ? "u16" : "u8",
                depth);
    std::printf("  Scalar decode_opt2:   %7.2f ms | %7.2f ns/op | %7.2f Mops/s\n",
                secs_scalar*1e3, (secs_scalar*1e9)/ops_scalar, ops_scalar/(secs_scalar*1e6));
    std::printf("  Batch  decode_batch:  %7.2f ms | %7.2f ns/op | %7.2f Mops/s\n",
                secs_batch*1e3, (secs_batch*1e9)/ops_batch, ops_batch/(secs_batch*1e6));
    std::printf("  correctness mismatches: %llu\n",
                (unsigned long long)mismatches);
}

template <typename CodeT>
static void run_suite(const char* code_label) {
    using DLC_ = Voxoctis::DepthLocationCode<CodeT, uint32_t>;
    struct DLC : public DLC_ {
        using code_type_alias = CodeT;
        using pos_type_alias  = uint32_t;
    };

    DLC dlc;

    // Choose a depth that fits the code width for the smoke tests
    constexpr uint8_t DMAX = max_depth_for_code<CodeT>();
    const uint8_t d = (DMAX >= 4) ? 4 : (DMAX ? DMAX : 1);
    const uint32_t mask = (d==0)?0u:((1u<<d)-1u);

    // Pick smoke coords that ALWAYS fit
    const uint32_t x0 = (1u) & mask;
    const uint32_t y0 = (2u) & mask;
    const uint32_t z0 = (3u) & mask;

    // --- Smoke tests (single case) ---
    auto k1 = dlc.encode_loop(x0, y0, z0, d);        auto a1 = dlc.decode_loop(k1);
    auto k2 = dlc.encode_magic(x0, y0, z0, d);       auto a2 = dlc.decode_magic(k2);
    auto k3 = dlc.encode_LUT(x0, y0, z0, d);         auto a3 = dlc.decode_LUT(k3);
    auto k4 = dlc.encode_BMI2(x0, y0, z0, d);        auto a4 = dlc.decode_BMI2(k4);
    auto k5 = dlc.encode_magic_opt(x0, y0, z0, d);   auto a5 = dlc.decode_magic(k5);
    auto k6 = dlc.encode_magic_opt2(x0, y0, z0, d);  auto a6 = dlc.decode_magic_opt2(k6);
    auto k7 = dlc.encode_magic_opt3(x0, y0, z0, d);  auto a7 = dlc.decode_magic_opt3(k7);
    auto k8 = dlc.encode_magic_opt2(x0, y0, z0, d);  auto a8 = dlc.decode_magic_opt2(k8); // AUTO-ish

    std::printf("\n== Smoke for codeType=%s, posType=uint32_t ==\n", code_label);
    std::printf("  d=%u  mask=0x%X  (x,y,z)=(%u,%u,%u)\n", (unsigned)d, mask, x0, y0, z0);
    std::printf("  smoke loop     : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a1), (unsigned)std::get<1>(a1), (unsigned)std::get<2>(a1), (unsigned long long)k1);
    std::printf("  smoke magic    : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a2), (unsigned)std::get<1>(a2), (unsigned)std::get<2>(a2), (unsigned long long)k2);
    std::printf("  smoke LUT      : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a3), (unsigned)std::get<1>(a3), (unsigned)std::get<2>(a3), (unsigned long long)k3);
    std::printf("  smoke BMI2     : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a4), (unsigned)std::get<1>(a4), (unsigned)std::get<2>(a4), (unsigned long long)k4);
    std::printf("  smoke magic_opt: x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a5), (unsigned)std::get<1>(a5), (unsigned)std::get<2>(a5), (unsigned long long)k5);
    std::printf("  smoke OPT2     : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a6), (unsigned)std::get<1>(a6), (unsigned)std::get<2>(a6), (unsigned long long)k6);
    std::printf("  smoke OPT3     : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a7), (unsigned)std::get<1>(a7), (unsigned)std::get<2>(a7), (unsigned long long)k7);
    std::printf("  smoke AUTO-ish : x=%u y=%u z=%u key=%llu\n",
        (unsigned)std::get<0>(a8), (unsigned)std::get<1>(a8), (unsigned)std::get<2>(a8), (unsigned long long)k8);

    // --- Accuracy pass (heavy) ---
    constexpr std::size_t kAccuracyN = 100000;
    accuracy_pass<DLC>(code_label, kAccuracyN);

    // --- Per-variant microbench ---
    static const char* variants[] = { "LOOP", "MAGIC", "LUT", "BMI2", "OPT", "OPT2", "OPT3", "AUTO" };
    static const std::size_t Ns[] = { 100, 1000, 1'000'000 };

    std::printf("\n== TIMING for codeType=%s (adaptive, >=50 ms per phase) ==\n", code_label);
    for (auto* v : variants) {
        for (auto N : Ns) {
            bench_variant<DLC>(v, N, /*min_ms=*/50.0);
        }
    }

    // --- Batch benches for ALL code widths (ALWAYS use dlc.encode_batch/decode_batch) ---
    const uint8_t batch_depth = DMAX ? DMAX : 1; // use the max valid depth for this key width
    bench_any_batch_encode<DLC>(1'000'000, batch_depth);
    bench_any_batch_decode<DLC>(1'000'000, batch_depth);
}

int main() {
#if defined(__AVX2__)
    std::printf("Build: __AVX2__ = yes\n");
#else
    std::printf("Build: __AVX2__ = no\n");
#endif
#if defined(__BMI2__)
    std::printf("Build: __BMI2__ = yes\n");
#else
    std::printf("Build: __BMI2__ = no\n");
#endif

    run_suite<uint8_t>("uint8_t");
    run_suite<uint16_t>("uint16_t");
    run_suite<uint32_t>("uint32_t");
    run_suite<uint64_t>("uint64_t");
    return 0;
}
