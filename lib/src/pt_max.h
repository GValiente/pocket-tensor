/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_MAX_H
#define PT_MAX_H

#include "pt_tensor.h"

namespace pt
{

struct ScalarMax
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; ++index)
        {
            *(r + index) = std::max(*(a + index), *(r + index));
        }
    }
};


struct VectorMax
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; index += Tensor::VectorSize)
        {
            Tensor::Vector av = simdpp::load(a + index);
            Tensor::Vector rv = simdpp::load(r + index);
            rv = simdpp::max(av, rv);
            simdpp::store(r + index, rv);
        }
    }
};


struct Vector2Max
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0, inc = Tensor::VectorSize; index != length; index += inc * 2)
        {
            Tensor::Vector av1 = simdpp::load(a + index);
            Tensor::Vector rv1 = simdpp::load(r + index);
            Tensor::Vector av2 = simdpp::load(a + index + inc);
            Tensor::Vector rv2 = simdpp::load(r + index + inc);
            rv1 = simdpp::max(av1, rv1);
            rv2 = simdpp::max(av2, rv2);
            simdpp::store(r + index, rv1);
            simdpp::store(r + index + inc, rv2);
        }
    }
};

}

#endif
