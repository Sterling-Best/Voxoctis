// depth_location_code.hpp
#pragma once

// depth_location_code.hpp
#ifndef VOXOCTIS_DEPTH_LOCATION_CODE_HPP
#define VOXOCTIS_DEPTH_LOCATION_CODE_HPP

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <array>

#if defined(_MSC_VER)
    #include <intrin.h>
    #define VOX_FORCEINLINE __forceinline
#else
    #define VOX_FORCEINLINE __attribute__((always_inline)) inline
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define VOX_LIKELY(x)   (__builtin_expect(!!(x), 1))
  #define VOX_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
  #define VOX_LIKELY(x)   (x)
  #define VOX_UNLIKELY(x) (x)
#endif


namespace Voxoctis {

template <typename codeType, typename posType>
class DepthLocationCode {

private:

    static constexpr uint64_t M3_0 = 0x1249249249249249ull;
    static constexpr uint64_t M3_1 = 0x10C30C30C30C30C3ull;
    static constexpr uint64_t M3_2 = 0x100F00F00F00F00Full;
    static constexpr uint64_t M3_3 = 0x1F0000FF0000FFull;
    static constexpr uint64_t M3_4 = 0x1F00000000FFFFull;
    static constexpr uint64_t M3_5 = 0x00000000001FFFFFull;

    using DLC64 = DepthLocationCode<uint64_t, uint32_t>;

    static constexpr uint8_t kDepthCap = uint8_t((sizeof(codeType)*8 - 1) / 3);

    inline static constexpr std::array<uint64_t,22> kDelim = []{
        std::array<uint64_t,22> a{};
        for (int i = 0; i <= 21; ++i) a[i] = 1ull << (3u * i);
        return a;
    }();

    inline static constexpr std::array<uint64_t,64> kClearDelimByMsb = []{
        std::array<uint64_t,64> a{};
        for (int m = 0; m < 64; ++m) {
            const uint8_t d = static_cast<uint8_t>(m / 3);     // same mapping as kMsbToDepth
            const uint64_t bit = (d < 22) ? (1ull << (3u * d)) : 0ull;
            a[m] = ~bit; // if you prefer AND; for XOR use just 'bit'
        }
        return a;
    }();

    // msb -> floor(msb/3) without division (msb in [0..63])
    inline static constexpr std::array<uint8_t,64> kMsbToDepth = []{
        std::array<uint8_t,64> a{};
        for (int i = 0; i < 64; ++i) a[i] = static_cast<uint8_t>(i / 3);
        return a;
    }();

    // Precomputed 8→24-bit lookup tables (compile-time; built once, no runtime cost)
    inline static constexpr std::array<uint32_t,256> ZLUT = []{
        std::array<uint32_t,256> t{};
        for (int b = 0; b < 256; ++b) {
            uint32_t v = 0;
            for (int i = 0; i < 8; ++i)
                if ((b >> i) & 1) v |= (1u << (3 * i)); // bits at 0,3,6,...
            t[b] = v;
        }
        return t;
    }();

    inline static constexpr std::array<uint32_t,256> YLUT = []{
        auto t = ZLUT;
        for (auto &x : t) x <<= 1;
        return t;
    }();

    inline static constexpr std::array<uint32_t,256> XLUT = []{
        auto t = ZLUT;
        for (auto &x : t) x <<= 2;
        return t;
    }();

    inline bool cpu_supports_avx2() {
    #if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int info[4] = {};
        __cpuid(info, 0);
        if (info[0] >= 7) {
            int info7[4] = {};
            __cpuidex(info7, 7, 0);
            // EBX bit 5
            return (info7[1] & (1 << 5)) != 0;
        }
        return false;
    #elif defined(__GNUC__) || defined(__clang__)
    #if defined(__x86_64__) && defined(__GLIBC__)
        return __builtin_cpu_supports("avx2");
    #else
        return false; // keep conservative on unknown C libraries/ABIs
    #endif
    #else
        return false;
    #endif
    }

    // Fast MSB index for 64-bit (BSR/LZCNT or fallback)
    static VOX_FORCEINLINE unsigned msb_index_u64(uint64_t v) {
    #if defined(_MSC_VER) && defined(_M_X64)
        unsigned long idx; _BitScanReverse64(&idx, v); return static_cast<unsigned>(idx);
    #elif defined(__GNUC__) || defined(__clang__)
        return 63u - static_cast<unsigned>(__builtin_clzll(v));
    #else
        unsigned msb = 0; for (uint64_t t = v; (t >>= 1) != 0; ) ++msb; return msb;
    #endif
    }

    template <typename T> struct use_lut_encode : std::bool_constant<(sizeof(T) <= 4)> {};

    #if defined(__AVX2__)
    static inline __m256i vox_const256_u64(uint64_t x){ return _mm256_set1_epi64x((long long)x); }

    #ifdef __AVX2__
    inline void encode_batch64_avx2(const uint32_t* x,
                                const uint32_t* y,
                                const uint32_t* z,
                                uint8_t depth,
                                uint64_t* out,
                                size_t n)
    {
        DLC64 dlc;
        // Uses your vector helper internally; handles the tail itself.
        dlc.encode_magic_opt2_avx2<uint32_t>(x, y, z, depth, out, n);
    }
    #endif

    inline void encode_batch64_fallback(const uint32_t* x,
                                    const uint32_t* y,
                                    const uint32_t* z,
                                    uint8_t depth,
                                    uint64_t* out,
                                    size_t n)
    {
        DLC64 dlc;
        for (size_t i = 0; i < n; ++i)
            out[i] = dlc.encode(x[i], y[i], z[i], depth);
    }

    #ifdef __AVX2__
    inline void decode_batch64_avx2(const uint64_t* keys,
                                uint32_t* x,
                                uint32_t* y,
                                uint32_t* z,
                                size_t n)
    {
        DLC64 dlc;
        dlc.decode_magic_opt2_avx2<uint32_t>(keys, x, y, z, n);
    }
    #endif

    inline void decode_batch64_fallback(const uint64_t* keys,
                                    uint32_t* x,
                                    uint32_t* y,
                                    uint32_t* z,
                                    size_t n)
    {
        DLC64 dlc;
        for (size_t i = 0; i < n; ++i) {
            auto t = dlc.decode(keys[i]);
            x[i] = std::get<0>(t);
            y[i] = std::get<1>(t);
            z[i] = std::get<2>(t);
        }
    }

    

    static inline void encode_magic_opt2_avx2_4(
        const uint32_t* x, const uint32_t* y, const uint32_t* z,
        uint8_t depth, uint64_t* out)
    {
        // Load as 64-bit lanes
        __m256i xv = _mm256_set_epi64x(x[3], x[2], x[1], x[0]);
        __m256i yv = _mm256_set_epi64x(y[3], y[2], y[1], y[0]);
        __m256i zv = _mm256_set_epi64x(z[3], z[2], z[1], z[0]);

        // (optional) pre-mask inputs to depth bits if you need strict masking
        if (depth < 21) {
            const __m256i m = _mm256_set1_epi64x((1ull << depth) - 1ull);
            xv = _mm256_and_si256(xv, m);
            yv = _mm256_and_si256(yv, m);
            zv = _mm256_and_si256(zv, m);
        }

        const __m256i X = spread3_21_avx2(xv);
        const __m256i Y = _mm256_slli_epi64(spread3_21_avx2(yv), 1);
        const __m256i Z = spread3_21_avx2(zv);

        __m256i morton = _mm256_or_si256(_mm256_slli_epi64(X, 2), _mm256_or_si256(Y, Z));

        // delimiter: (1ull << (3*depth)) is uniform; broadcast then OR
        const __m256i delim = _mm256_set1_epi64x(1ull << (3u * depth));
        morton = _mm256_or_si256(morton, delim);

        // Store (note: elements are reversed relative to set_epi64x order)
        alignas(32) uint64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, morton);
        out[0] = tmp[0];
        out[1] = tmp[1];
        out[2] = tmp[2];
        out[3] = tmp[3];
    }

    

    //----Encode/Decode Helper Functions
    static VOX_FORCEINLINE uint64_t compact3_u64(uint64_t v) {
        v &= 0x1249249249249249ull;
        v = (v ^ (v >>  2)) & 0x10C30C30C30C30C3ull;
        v = (v ^ (v >>  4)) & 0x100F00F00F00F00Full;
        v = (v ^ (v >>  8)) & 0x1F0000FF0000FFull;
        v = (v ^ (v >> 16)) & 0x1F00000000FFFFull;
        v = (v ^ (v >> 32)) & 0x1FFFFFull;
        return v;
    }

    // 32-bit interleave (up to 10 bits) — fewer masks/shifts than 64-bit version
    static VOX_FORCEINLINE uint32_t spread3_10(uint32_t v) {
        v &= 0x3FFu;                         // 10 bits
        v = (v | (v << 16)) & 0x030000FFu;   // 00000011 00000000 00000000 11111111
        v = (v | (v <<  8)) & 0x0300F00Fu;   // 00000011 00000000 11110000 00001111
        v = (v | (v <<  4)) & 0x030C30C3u;   // 00000011 00001100 00110000 11000011
        v = (v | (v <<  2)) & 0x09249249u;   // 00001001 00100100 10010010 01001001
        return v;
    }

    static VOX_FORCEINLINE uint32_t compact3_10(uint32_t v) {
        v &= 0x09249249u;
        v = (v ^ (v >>  2)) & 0x030C30C3u;
        v = (v ^ (v >>  4)) & 0x0300F00Fu;
        v = (v ^ (v >>  8)) & 0x030000FFu;
        v = (v ^ (v >> 16)) & 0x000003FFu;
        return v;
    }

    // 64-bit interleave (up to 21 bits)
    static VOX_FORCEINLINE uint64_t spread3_21(uint64_t v) {
        v &= 0x1FFFFFu;                        // 21 bits
        v = (v | (v << 32)) & 0x1F00000000FFFFull;
        v = (v | (v << 16)) & 0x1F0000FF0000FFull;
        v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
        v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
        v = (v | (v <<  2)) & 0x1249249249249249ull;
        return v;
    }

    static VOX_FORCEINLINE uint64_t compact3_21(uint64_t v) {
        v &= 0x1249249249249249ull;
        v = (v ^ (v >>  2)) & 0x10C30C30C30C30C3ull;
        v = (v ^ (v >>  4)) & 0x100F00F00F00F00Full;
        v = (v ^ (v >>  8)) & 0x1F0000FF0000FFull;
        v = (v ^ (v >> 16)) & 0x1F00000000FFFFull;
        v = (v ^ (v >> 32)) & 0x00000000001FFFFFull;
        return v;
    }

    static inline __m256i spread3_21_avx2(__m256i v) {
        const __m256i m21  = _mm256_set1_epi64x(0x1FFFFFull);
        const __m256i m32  = _mm256_set1_epi64x(0x1F00000000FFFFull);
        const __m256i m16  = _mm256_set1_epi64x(0x1F0000FF0000FFull);
        const __m256i m8   = _mm256_set1_epi64x(0x100F00F00F00F00Full);
        const __m256i m4   = _mm256_set1_epi64x(0x10C30C30C30C30C3ull);
        const __m256i m2   = _mm256_set1_epi64x(0x1249249249249249ull);

        __m256i t = _mm256_and_si256(v, m21);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t, 32)), m32);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t, 16)), m16);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  8)), m8 );
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  4)), m4 );
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  2)), m2 );
        return t;
    }

    static inline void spread3_21_avx2_inplace(__m256i& t) {
        const __m256i m21 = vox_const256_u64(0x1FFFFFull);
        const __m256i m32 = vox_const256_u64(0x1F00000000FFFFull);
        const __m256i m16 = vox_const256_u64(0x1F0000FF0000FFull);
        const __m256i m8  = vox_const256_u64(0x100F00F00F00F00Full);
        const __m256i m4  = vox_const256_u64(0x10C30C30C30C30C3ull);
        const __m256i m2  = vox_const256_u64(0x1249249249249249ull);

        t = _mm256_and_si256(t, m21);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t, 32)), m32);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t, 16)), m16);
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  8)), m8 );
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  4)), m4 );
        t = _mm256_and_si256(_mm256_or_si256(t, _mm256_slli_epi64(t,  2)), m2 );
    }

    #if defined(__AVX2__)
    static inline __m256i compact3_21_avx2(__m256i v) {
        const __m256i m0 = _mm256_set1_epi64x(0x1249249249249249ull);
        const __m256i m1 = _mm256_set1_epi64x(0x10C30C30C30C30C3ull);
        const __m256i m2 = _mm256_set1_epi64x(0x100F00F00F00F00Full);
        const __m256i m3 = _mm256_set1_epi64x(0x001F0000FF0000FFull);
        const __m256i m4 = _mm256_set1_epi64x(0x001F00000000FFFFull);
        const __m256i m5 = _mm256_set1_epi64x(0x00000000001FFFFFull);

        __m256i t = _mm256_and_si256(v, m0);
        t = _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi64(t,  2)), m1);
        t = _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi64(t,  4)), m2);
        t = _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi64(t,  8)), m3);
        t = _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi64(t, 16)), m4);
        t = _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi64(t, 32)), m5);
        return t;
    }


public:

    static_assert(std::is_unsigned<codeType>::value, "codeType must be unsigned");
    static_assert(std::is_unsigned<posType>::value,  "posType must be unsigned");

    // NEW (non-static; calls non-static members just fine)
    VOX_FORCEINLINE codeType encode(posType x,posType y,posType z,uint8_t depth) {
        if constexpr (use_lut_encode<codeType>::value) return encode_LUT(x,y,z,depth);
        else                                           return encode_magic_opt2(x,y,z,depth);
    }

    VOX_FORCEINLINE std::tuple<posType,posType,posType> decode(codeType key) {
        return decode_magic_opt2(key);
    }

    // ----------------------- ENCODE (batch) -----------------------
    
    inline void encode_batch(const posType* __restrict x,
                         const posType* __restrict y,
                         const posType* __restrict z,
                         uint8_t depth,
                         codeType* __restrict out,
                         size_t n)
    {
        if constexpr (sizeof(codeType) == 8) {
            // 64-bit: use your existing vector path
        #if defined(__AVX2__)
            if (cpu_supports_avx2()) {
                // Inputs are posType (usually uint32_t). Safe to reinterpret to the helper’s uint32_t*.
                encode_batch64_avx2(reinterpret_cast<const uint32_t*>(x),
                                    reinterpret_cast<const uint32_t*>(y),
                                    reinterpret_cast<const uint32_t*>(z),
                                    depth,
                                    reinterpret_cast<uint64_t*>(out),
                                    n);
                return;
            }
        #endif
            encode_batch64_fallback(reinterpret_cast<const uint32_t*>(x),
                                    reinterpret_cast<const uint32_t*>(y),
                                    reinterpret_cast<const uint32_t*>(z),
                                    depth,
                                    reinterpret_cast<uint64_t*>(out),
                                    n);
        } else {
            // ≤32-bit: fast LUT path (depth ≤ 10 by construction for 8/16/32-bit codes)
            for (size_t i = 0; i < n; ++i)
                out[i] = encode_LUT(x[i], y[i], z[i], depth);
        }
    }

    template<class Pos=uint32_t>
    VOX_FORCEINLINE void encode_magic_opt2_avx2(
        const Pos* __restrict x,
        const Pos* __restrict y,
        const Pos* __restrict z,
        uint8_t depth,
        uint64_t* __restrict out,
        size_t n)
    {
        const __m256i delim = _mm256_set1_epi64x(1ull << (3u*depth));
        size_t i = 0;

        // Process 8 at a time to reduce loop overhead
        for (; i + 7 < n; i += 8) {
            __m256i xv0 = _mm256_set_epi64x(x[i+3], x[i+2], x[i+1], x[i+0]);
            __m256i yv0 = _mm256_set_epi64x(y[i+3], y[i+2], y[i+1], y[i+0]);
            __m256i zv0 = _mm256_set_epi64x(z[i+3], z[i+2], z[i+1], z[i+0]);

            __m256i xv1 = _mm256_set_epi64x(x[i+7], x[i+6], x[i+5], x[i+4]);
            __m256i yv1 = _mm256_set_epi64x(y[i+7], y[i+6], y[i+5], y[i+4]);
            __m256i zv1 = _mm256_set_epi64x(z[i+7], z[i+6], z[i+5], z[i+4]);

            spread3_21_avx2_inplace(xv0);
            spread3_21_avx2_inplace(yv0);
            spread3_21_avx2_inplace(zv0);
            __m256i m0 = _mm256_or_si256(_mm256_slli_epi64(xv0, 2),
                    _mm256_or_si256(_mm256_slli_epi64(yv0, 1), zv0));
            m0 = _mm256_or_si256(m0, delim);

            spread3_21_avx2_inplace(xv1);
            spread3_21_avx2_inplace(yv1);
            spread3_21_avx2_inplace(zv1);
            __m256i m1 = _mm256_or_si256(_mm256_slli_epi64(xv1, 2),
                    _mm256_or_si256(_mm256_slli_epi64(yv1, 1), zv1));
            m1 = _mm256_or_si256(m1, delim);

            _mm256_storeu_si256((__m256i*)(out + i + 0), m0);
            _mm256_storeu_si256((__m256i*)(out + i + 4), m1);
        }

        // tail, scalar
        for (; i < n; ++i) {
            const uint64_t X = spread3_21((uint64_t)x[i]);
            const uint64_t Y = spread3_21((uint64_t)y[i]) << 1;
            const uint64_t Z = spread3_21((uint64_t)z[i]);
            out[i] = (X<<2) | Y | Z | (1ull << (3u*depth));
        }
    }

    


    // ----------------------- DECODE (batch) -----------------------
    

    // Decodes arrays of keys[] of type codeType -> (x[],y[],z[]).
    // 64-bit uses AVX2/compact3_21 path; ≤32-bit uses the small-depth decoder.
    inline void decode_batch(const codeType* __restrict keys,
                            posType* __restrict x,
                            posType* __restrict y,
                            posType* __restrict z,
                            size_t n)
    {
        if constexpr (sizeof(codeType) == 8) {
        #if defined(__AVX2__)
            if (cpu_supports_avx2()) {
                decode_batch64_avx2(reinterpret_cast<const uint64_t*>(keys),
                                    reinterpret_cast<uint32_t*>(x),
                                    reinterpret_cast<uint32_t*>(y),
                                    reinterpret_cast<uint32_t*>(z),
                                    n);
                return;
            }
        #endif
            decode_batch64_fallback(reinterpret_cast<const uint64_t*>(keys),
                                    reinterpret_cast<uint32_t*>(x),
                                    reinterpret_cast<uint32_t*>(y),
                                    reinterpret_cast<uint32_t*>(z),
                                    n);
        } else {
            // ≤32-bit keys: small-depth path via decode_magic_opt2 (uses compact3_10 internally)
            for (size_t i = 0; i < n; ++i) {
                auto t = decode_magic_opt2(keys[i]);
                x[i] = std::get<0>(t);
                y[i] = std::get<1>(t);
                z[i] = std::get<2>(t);
            }
        }
    }

    template<class Pos=uint32_t>
    void decode_magic_opt2_avx2(
        const uint64_t* keys, Pos* x, Pos* y, Pos* z, size_t n)
    {
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            // Scalar depth/MSB + delimiter clear (cheap)
            uint64_t p[4];
            for (int k=0; k<4; ++k) {
                uint64_t ku = keys[i+k];
                unsigned msb;
            #if defined(_MSC_VER) && defined(_M_X64)
                unsigned long idx; _BitScanReverse64(&idx, ku); msb = (unsigned)idx;
            #else
                msb = 63u - (unsigned)__builtin_clzll(ku);
            #endif
                uint8_t depth = static_cast<uint8_t>(msb / 3);
                p[k] = ku & ~(1ull << (3u * depth));
            }

            // Vectorize the heavy compact phases for x/y/z
            __m256i P = _mm256_set_epi64x(p[3], p[2], p[1], p[0]);
            __m256i X = compact3_21_avx2(_mm256_srli_epi64(P, 2));
            __m256i Y = compact3_21_avx2(_mm256_srli_epi64(P, 1));
            __m256i Z = compact3_21_avx2(P);

            alignas(32) uint64_t tx[4], ty[4], tz[4];
            _mm256_store_si256((__m256i*)tx, X);
            _mm256_store_si256((__m256i*)ty, Y);
            _mm256_store_si256((__m256i*)tz, Z);

            x[i+0] = (Pos)tx[0]; x[i+1] = (Pos)tx[1]; x[i+2] = (Pos)tx[2]; x[i+3] = (Pos)tx[3];
            y[i+0] = (Pos)ty[0]; y[i+1] = (Pos)ty[1]; y[i+2] = (Pos)ty[2]; y[i+3] = (Pos)ty[3];
            z[i+0] = (Pos)tz[0]; z[i+1] = (Pos)tz[1]; z[i+2] = (Pos)tz[2]; z[i+3] = (Pos)tz[3];
        }

        // tail
        for (; i < n; ++i) {
            const uint64_t ku = keys[i];
            unsigned msb;
        #if defined(_MSC_VER) && defined(_M_X64)
            unsigned long idx; _BitScanReverse64(&idx, ku); msb = (unsigned)idx;
        #else
            msb = 63u - (unsigned)__builtin_clzll(ku);
        #endif
            uint8_t depth = (uint8_t)(msb / 3);
            const uint64_t p = ku & ~(1ull << (3u * depth));
            x[i] = (Pos)Voxoctis::DepthLocationCode<uint64_t,Pos>::compact3_21(p >> 2);
            y[i] = (Pos)Voxoctis::DepthLocationCode<uint64_t,Pos>::compact3_21(p >> 1);
            z[i] = (Pos)Voxoctis::DepthLocationCode<uint64_t,Pos>::compact3_21(p >> 0);
        }
    }      

    VOX_FORCEINLINE codeType encode_magic_opt2(posType x, posType y, posType z, uint8_t depth) {
        const uint64_t X = spread3_21(static_cast<uint64_t>(x));
        const uint64_t Y = spread3_21(static_cast<uint64_t>(y)) << 1;
        const uint64_t Z = spread3_21(static_cast<uint64_t>(z));
        const uint64_t morton = (X << 2) | Y | Z;
        return static_cast<codeType>(morton | (1ull << (3u * depth)));
    }

    VOX_FORCEINLINE std::tuple<posType,posType,posType>decode_magic_opt2(codeType key) {
        const uint64_t ku  = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint64_t pay = ku & kClearDelimByMsb[msb];
        const uint8_t d = kMsbToDepth[msb];

        if constexpr (sizeof(codeType) <= 4) {
            // Compile-time: only the <=10 path exists, so no runtime branch here.
            const uint32_t p = static_cast<uint32_t>(pay);
            const posType x  = static_cast<posType>(compact3_10(p >> 2));
            const posType y  = static_cast<posType>(compact3_10(p >> 1));
            const posType z  = static_cast<posType>(compact3_10(p >> 0));
            return {x,y,z};
        } else {
            // 64-bit code: keep the small-depth path (common) as the likely path.
            if (VOX_LIKELY( d <= 10)) {
                const uint32_t p = static_cast<uint32_t>(pay);
                const posType x  = static_cast<posType>(compact3_10(p >> 2));
                const posType y  = static_cast<posType>(compact3_10(p >> 1));
                const posType z  = static_cast<posType>(compact3_10(p >> 0));
                return {x,y,z};
            } else {
                const posType x  = static_cast<posType>(compact3_21(pay >> 2));
                const posType y  = static_cast<posType>(compact3_21(pay >> 1));
                const posType z  = static_cast<posType>(compact3_21(pay >> 0));
                return {x,y,z};
            }
        }
    
    }

    VOX_FORCEINLINE codeType encode_LUT(posType x, posType y, posType z, uint8_t depth) {
        const unsigned bytes = (depth + 7u) / 8u; // only process needed bytes
        uint64_t morton = 0;
        for (unsigned byte = 0; byte < bytes; ++byte) {
            const uint8_t xb = uint8_t((x >> (8u*byte)) & 0xFFu);
            const uint8_t yb = uint8_t((y >> (8u*byte)) & 0xFFu);
            const uint8_t zb = uint8_t((z >> (8u*byte)) & 0xFFu);
            const uint32_t m = XLUT[xb] | YLUT[yb] | ZLUT[zb];
            morton |= (m << (24u * byte));
        }
        return static_cast<codeType>(morton | kDelim[depth]);
    }
    
    //----Decommissioned Functions
     codeType encode_magic_opt3(posType x, posType y, posType z, uint8_t depth) {
        // Pre-mask
        const unsigned pw = sizeof(posType) * 8u;
        const posType m = (depth >= pw) ? ~posType(0) : (posType(1) << depth) - 1;
        x &= m; y &= m; z &= m;

        if (depth <= 10) {
            uint32_t X = spread3_10(static_cast<uint32_t>(x));
            uint32_t Y = spread3_10(static_cast<uint32_t>(y)) << 1;
            uint32_t Z = spread3_10(static_cast<uint32_t>(z));
            uint32_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(uint64_t(morton) | (uint64_t(1) << (3u * depth)));
        } else {
            uint64_t X = spread3_21((uint64_t)x);
            uint64_t Y = spread3_21((uint64_t)y) << 1;
            uint64_t Z = spread3_21((uint64_t)z);
            uint64_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(morton | (uint64_t(1) << (3u * depth)));
        }
    }

    std::tuple<posType,posType,posType> decode_magic_opt3(codeType key) {
        if (!key) return {posType(0),posType(0),posType(0)};
        const uint64_t ku = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint8_t depth = kMsbToDepth[msb];
        const uint64_t payload = ku & ~(1ull << (3u * depth));

        if (depth <= 10) {
            const uint32_t p = static_cast<uint32_t>(payload);
            const posType x = static_cast<posType>(compact3_10(p >> 2));
            const posType y = static_cast<posType>(compact3_10(p >> 1));
            const posType z = static_cast<posType>(compact3_10(p >> 0));
            return {x,y,z};
        } else {
            const posType x = static_cast<posType>(compact3_21(payload >> 2));
            const posType y = static_cast<posType>(compact3_21(payload >> 1));
            const posType z = static_cast<posType>(compact3_21(payload >> 0));
            return {x,y,z};
        }
    }

    codeType encode_magic_opt(posType x, posType y, posType z, uint8_t depth) {
        // Pre-mask
        const unsigned pw = sizeof(posType) * 8u;
        const posType m = (depth >= pw) ? ~posType(0) : (posType(1) << depth) - 1;
        x &= m; y &= m; z &= m;

        if (depth <= 10) {
            uint32_t X = spread3_10(static_cast<uint32_t>(x));
            uint32_t Y = spread3_10(static_cast<uint32_t>(y)) << 1;
            uint32_t Z = spread3_10(static_cast<uint32_t>(z));
            uint32_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(uint64_t(morton) | (uint64_t(1) << (3u * depth)));
        } else {
            uint64_t X = spread3_21(static_cast<uint64_t>(x));
            uint64_t Y = spread3_21(static_cast<uint64_t>(y)) << 1;
            uint64_t Z = spread3_21(static_cast<uint64_t>(z));
            uint64_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(morton | (uint64_t(1) << (3u * depth)));
        }
    }

    codeType encode_magic(posType x, posType y, posType z, uint8_t depth) {
        // Pre-mask
        const unsigned pw = sizeof(posType) * 8u;
        const posType m = (depth >= pw) ? ~posType(0) : (posType(1) << depth) - 1;
        x &= m; y &= m; z &= m;

        if (depth <= 10) {
            uint32_t X = spread3_10(static_cast<uint32_t>(x));
            uint32_t Y = spread3_10(static_cast<uint32_t>(y)) << 1;
            uint32_t Z = spread3_10(static_cast<uint32_t>(z));
            uint32_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(uint64_t(morton) | (uint64_t(1) << (3u * depth)));
        } else {
            auto spread3 = [](uint64_t v) -> uint64_t {
          
                v &= 0x1FFFFFu; // lower 21 bits
                v = (v | (v << 32)) & 0x1F00000000FFFFull;
                v = (v | (v << 16)) & 0x1F0000FF0000FFull;
                v = (v | (v << 8))  & 0x100F00F00F00F00Full;
                v = (v | (v << 4))  & 0x10C30C30C30C30C3ull;
                v = (v | (v << 2))  & 0x1249249249249249ull;
                return v;
            };

            uint64_t X = spread3(static_cast<uint64_t>(x));
            uint64_t Y = spread3(static_cast<uint64_t>(y)) << 1;
            uint64_t Z = spread3(static_cast<uint64_t>(z));
            uint64_t morton = (X << 2) | Y | Z;
            return static_cast<codeType>(morton | (1ull << (3u * depth)));
        }
    }

     std::tuple<posType,posType,posType> decode_magic(codeType key) {
        if (key == 0) return { posType(0), posType(0), posType(0) };

        const uint64_t ku = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint8_t depth = kMsbToDepth[msb];
        const uint64_t payload = ku & ~(1ull << (3u * depth));

        if (depth <= 10) {
            const uint32_t p = static_cast<uint32_t>(payload);
            posType x = static_cast<posType>(compact3_10(p >> 2));
            posType y = static_cast<posType>(compact3_10(p >> 1));
            posType z = static_cast<posType>(compact3_10(p >> 0));
            return { x, y, z };
        } else {
            auto compact3 = [](uint64_t v) -> uint64_t {
                v &= 0x1249249249249249ull;
                v = (v ^ (v >> 2)) & 0x10C30C30C30C30C3ull;
                v = (v ^ (v >> 4)) & 0x100F00F00F00F00Full;
                v = (v ^ (v >> 8)) & 0x1F0000FF0000FFull;
                v = (v ^ (v >> 16)) & 0x1F00000000FFFFull;
                v = (v ^ (v >> 32)) & 0x1FFFFFull;
                return v;
            };
            posType x = static_cast<posType>(compact3(payload >> 2));
            posType y = static_cast<posType>(compact3(payload >> 1));
            posType z = static_cast<posType>(compact3(payload >> 0));
            return { x, y, z };
        }
    }

    codeType encode_BMI2(posType x, posType y, posType z, uint8_t depth) {
    #if defined(__BMI2__) || (defined(_MSC_VER) && defined(__AVX2__))
        // Interleave using PDEP and add delimiter at bit 3*depth.
        constexpr uint64_t M64 = 0x1249249249249249ull; // bits 0,3,6,...
        constexpr uint32_t M32 = 0x09249249u;          // bits 0,3,6,... (32-bit)

        // Mask coordinates to 'depth' bits first.
        const unsigned pw = sizeof(posType) * 8u;
        if (depth < pw) {
            const posType m = (posType(1) << depth) - 1;
            x &= m; y &= m; z &= m;
        }

        if constexpr (sizeof(codeType) <= 4) {
            uint32_t X = _pdep_u32(static_cast<uint32_t>(x), M32) << 2;
            uint32_t Y = _pdep_u32(static_cast<uint32_t>(y), M32) << 1;
            uint32_t Z = _pdep_u32(static_cast<uint32_t>(z), M32);
            uint32_t morton = X | Y | Z;
            morton |= (uint32_t(1) << (3u * depth));
            return static_cast<codeType>(morton);
        } else {
            uint64_t X = _pdep_u64(static_cast<uint64_t>(x), M64) << 2;
            uint64_t Y = _pdep_u64(static_cast<uint64_t>(y), M64) << 1;
            uint64_t Z = _pdep_u64(static_cast<uint64_t>(z), M64);
            uint64_t morton = X | Y | Z;
            morton |= (uint64_t(1) << (3u * depth));
            return static_cast<codeType>(morton);
        }
    #else
        // Fallback if BMI2 not enabled at compile-time.
        return encode_magic_opt2(x, y, z, depth);
    #endif
    }

    std::tuple<posType, posType, posType> decode_BMI2(codeType key) {
    #if defined(__BMI2__) || (defined(_MSC_VER) && defined(__AVX2__))
        if (key == codeType(0)) return { posType(0), posType(0), posType(0) };

        // Fast depth + AND-clear
        const uint64_t ku = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint8_t depth = kMsbToDepth[msb];
        const uint64_t payload = ku & ~(1ull << (3u * depth));

        constexpr uint64_t M64 = 0x1249249249249249ull;
        constexpr uint32_t M32 = 0x09249249u;

        if constexpr (sizeof(codeType) <= 4) {
            uint32_t p = static_cast<uint32_t>(payload);
            posType x = static_cast<posType>(_pext_u32(p >> 2, M32));
            posType y = static_cast<posType>(_pext_u32(p >> 1, M32));
            posType z = static_cast<posType>(_pext_u32(p >> 0, M32));
            return { x, y, z };
        } else {
            uint64_t p = static_cast<uint64_t>(payload);
            posType x = static_cast<posType>(_pext_u64(p >> 2, M64));
            posType y = static_cast<posType>(_pext_u64(p >> 1, M64));
            posType z = static_cast<posType>(_pext_u64(p >> 0, M64));
            return { x, y, z };
        }
    #else
        // Fallback if BMI2 not enabled at compile-time.
        return decode_magic_opt2(key);
    #endif
    }

    codeType encode_loop(posType x, posType y, posType z, uint8_t depth) {
        // Pre-mask
        const unsigned pw = sizeof(posType) * 8u;
        const posType m = (depth >= pw) ? ~posType(0) : (posType(1) << depth) - 1;
        x &= m; y &= m; z &= m;

        codeType out = 0;
        for (uint8_t i = 0; i < depth; ++i) {
            codeType xb = codeType((x >> i) & 1u) << (3u * i + 2u);
            codeType yb = codeType((y >> i) & 1u) << (3u * i + 1u);
            codeType zb = codeType((z >> i) & 1u) << (3u * i + 0u);
            out |= (xb | yb | zb);
        }
        codeType delimiter = codeType(1) << (3u * depth);
        return delimiter | out;
    }

    std::tuple<posType, posType, posType> decode_loop(codeType key) {
        if (key == codeType(0)) {
            return { posType(0), posType(0), posType(0) };
        }
        // fast depth
        const uint64_t ku = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint8_t depth = kMsbToDepth[msb];
        const codeType payload = static_cast<codeType>(ku & ~(1ull << (3u * depth)));

        posType x = 0, y = 0, z = 0;
        for (uint8_t i = 0; i < depth; ++i) {
            codeType tri = (payload >> (3u * i)) & codeType(7);
            x |= posType((tri >> 2) & 1u) << i;
            y |= posType((tri >> 1) & 1u) << i;
            z |= posType((tri >> 0) & 1u) << i;
        }
        return { x, y, z };
    }

     std::tuple<posType, posType, posType> decode_LUT(codeType key) {
        if (key == 0) return { posType(0), posType(0), posType(0) };

        // Fast depth; AND-clear delimiter
        const uint64_t ku = static_cast<uint64_t>(key);
        const unsigned msb = msb_index_u64(ku);
        const uint8_t depth = kMsbToDepth[msb];
        const uint64_t payload = ku & ~(1ull << (3u*depth));

        posType x = 0, y = 0, z = 0;
        const unsigned bytes = (depth + 7u) / 8u;
        for (unsigned byte = 0; byte < bytes; ++byte) {
            const uint32_t chunk = uint32_t((payload >> (24u*byte)) & 0xFFFFFFu);
            // Use the faster 32-bit compactor; each 24-bit chunk holds <= 8 bits/lane
            const uint8_t xb = uint8_t(compact3_10((chunk >> 2)));
            const uint8_t yb = uint8_t(compact3_10((chunk >> 1)));
            const uint8_t zb = uint8_t(compact3_10((chunk >> 0)));

            x |= posType(xb) << (8u*byte);
            y |= posType(yb) << (8u*byte);
            z |= posType(zb) << (8u*byte);
        }

        const unsigned pw   = sizeof(posType) * 8u;
        const posType mask  = (depth >= pw) ? ~posType(0) : (posType(1) << depth) - 1;
        return { posType(x & mask), posType(y & mask), posType(z & mask) };
    }



};

#undef VOX_FORCEINLINE


} // namespace Voxoctis

#endif
#endif
#endif