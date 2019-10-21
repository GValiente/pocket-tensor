/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LIBSIMDPP_H
#define PT_LIBSIMDPP_H

#define SIMDPP_DISABLE_DEPRECATED_IN_2_1_AND_OLDER 1

#include "pt_tweakme.h"

#if defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align"
    #pragma GCC diagnostic ignored "-Wcast-qual"
    #pragma GCC diagnostic ignored "-Wfloat-equal"
#elif defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4244)
#endif

#include "simdpp/simd.h"

#if defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

#define PT_INLINE SIMDPP_INL

namespace pt
{
    #if PT_DOUBLE_ENABLE
        using FloatType = double;
        using FloatVector = simdpp::float64v;
        static constexpr auto FloatSize = SIMDPP_FAST_FLOAT64_SIZE;
    #else
        using FloatType = float;
        using FloatVector = simdpp::float32v;
        static constexpr auto FloatSize = SIMDPP_FAST_FLOAT32_SIZE;
    #endif

    template<typename T>
    PT_INLINE simdpp::expr_vec_make_const<T, 1> makeVector(T value) noexcept
    {
        simdpp::expr_vec_make_const<T, 1> a;
        a.a[0] = value;
        return a;
    }
}

#endif
