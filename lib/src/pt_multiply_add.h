/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_MULTIPLY_ADD_H
#define PT_MULTIPLY_ADD_H

#include "pt_tensor.h"

namespace pt
{

namespace detail
{
    PT_INLINE Tensor::Vector madd(const Tensor::Vector& av, const Tensor::Vector& bv,
                                      const Tensor::Vector& rv) noexcept
    {
        #if PT_FMADD_ENABLE
            return simdpp::fmadd(av, bv, rv);
        #else
            return simdpp::add(rv, simdpp::mul(av, bv));
        #endif
    }
}

struct ScalarMultiplyAdd
{
    PT_INLINE void operator()(const Tensor::Type* a, const Tensor::Type* b, Tensor::Type* r,
                                  int length) noexcept
    {
        for(int index = 0; index != length; ++index)
        {
            *(r + index) += *(a + index) * *(b  + index);
        }
    }

    PT_INLINE Tensor::Type operator()(const Tensor::Type* a, const Tensor::Type* b,
                                          int length) noexcept
    {
        Tensor::Type r(0);

        for(int index = 0; index != length; ++index)
        {
            r += *(a + index) * *(b  + index);
        }

        return r;
    }
};


struct VectorMultiplyAdd
{
    PT_INLINE void operator()(const Tensor::Type* a, const Tensor::Type* b, Tensor::Type* r,
                                  int length) noexcept
    {
        for(int index = 0; index != length; index += Tensor::VectorSize)
        {
            Tensor::Vector rv = simdpp::load(r + index);
            rv = detail::madd(simdpp::load(a + index), simdpp::load(b + index), rv);
            simdpp::store(r + index, rv);
        }
    }

    PT_INLINE Tensor::Type operator()(const Tensor::Type* a, const Tensor::Type* b,
                                          int length) noexcept
    {
        Tensor::Vector rv = makeVector(Tensor::Type(0));

        for(int index = 0; index != length; index += Tensor::VectorSize)
        {
            rv = detail::madd(simdpp::load(a + index), simdpp::load(b + index), rv);
        }

        return simdpp::reduce_add(rv);
    }
};


struct Vector2MultiplyAdd
{
    PT_INLINE void operator()(const Tensor::Type* a, const Tensor::Type* b, Tensor::Type* r,
                                  int length) noexcept
    {
        for(int index = 0, inc = Tensor::VectorSize; index != length; index += inc * 2)
        {
            Tensor::Vector rv1 = simdpp::load(r + index);
            Tensor::Vector rv2 = simdpp::load(r + index + inc);
            rv1 = detail::madd(simdpp::load(a + index), simdpp::load(b + index), rv1);
            rv2 = detail::madd(simdpp::load(a + index + inc), simdpp::load(b + index + inc), rv2);
            simdpp::store(r + index, rv1);
            simdpp::store(r + index + inc, rv2);
        }
    }

    PT_INLINE Tensor::Type operator()(const Tensor::Type* a, const Tensor::Type* b,
                                          int length) noexcept
    {
        Tensor::Vector rv1 = makeVector(Tensor::Type(0));
        Tensor::Vector rv2 = makeVector(Tensor::Type(0));

        for(int index = 0, inc = Tensor::VectorSize; index != length; index += inc * 2)
        {
            rv1 = detail::madd(simdpp::load(a + index), simdpp::load(b + index), rv1);
            rv2 = detail::madd(simdpp::load(a + index + inc), simdpp::load(b + index + inc), rv2);
        }

        return simdpp::reduce_add(rv1) + simdpp::reduce_add(rv2);
    }
};

}

#endif
