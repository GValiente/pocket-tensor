/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_ADD_H
#define PT_ADD_H

#include "pt_tensor.h"

namespace pt
{

struct ScalarAdd
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; ++index)
        {
            *(r + index) += *(a + index);
        }
    }
};


struct VectorAdd
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0; index != length; index += Tensor::VectorSize)
        {
            Tensor::Vector av = simdpp::load(a + index);
            Tensor::Vector rv = simdpp::load(r + index);
            rv = simdpp::add(av, rv);
            simdpp::store(r + index, rv);
        }
    }
};


struct Vector2Add
{
    PT_INLINE void operator()(const Tensor::Type* a, Tensor::Type* r, int length) noexcept
    {
        for(int index = 0, inc = Tensor::VectorSize; index != length; index += inc * 2)
        {
            Tensor::Vector av1 = simdpp::load(a + index);
            Tensor::Vector rv1 = simdpp::load(r + index);
            Tensor::Vector av2 = simdpp::load(a + index + inc);
            Tensor::Vector rv2 = simdpp::load(r + index + inc);
            rv1 = simdpp::add(av1, rv1);
            rv2 = simdpp::add(av2, rv2);
            simdpp::store(r + index, rv1);
            simdpp::store(r + index + inc, rv2);
        }
    }
};

}

#endif
