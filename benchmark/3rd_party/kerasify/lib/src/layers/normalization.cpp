/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

bool BatchNormalization::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file));
    check(biases_.load(file));
    return true;
}

bool BatchNormalization::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() > 0);
    out = in.fma(weights_, biases_);
    return true;
}

} // namespace layers
} // namespace keras
