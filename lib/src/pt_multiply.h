/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_MULTIPLY_H
#define PT_MULTIPLY_H

#include "pt_tensor.h"

namespace pt
{

struct ScalarMultiply
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; ++index)
        {
            *(r + index) *= *(a + index);
        }
    }
};


struct VectorMultiply
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; index += Tensor::VectorSize)
        {
            Tensor::Vector av = simdpp::load(a + index);
            Tensor::Vector rv = simdpp::load(r + index);
            rv = simdpp::mul(av, rv);
            simdpp::store(r + index, rv);
        }
    }
};


struct Vector2Multiply
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0, inc = Tensor::VectorSize; index != length; index += inc * 2)
        {
            Tensor::Vector av1 = simdpp::load(a + index);
            Tensor::Vector rv1 = simdpp::load(r + index);
            Tensor::Vector av2 = simdpp::load(a + index + inc);
            Tensor::Vector rv2 = simdpp::load(r + index + inc);
            rv1 = simdpp::mul(av1, rv1);
            rv2 = simdpp::mul(av2, rv2);
            simdpp::store(r + index, rv1);
            simdpp::store(r + index + inc, rv2);
        }
    }
};

}

#endif
