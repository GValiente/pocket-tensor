/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/dense.h"

namespace keras {
namespace layers {

bool Dense::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file, 2));
    check(biases_.load(file));
    check(activation_.load_layer(file));
    return true;
}

bool Dense::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.size() == weights_.dims_[1]);

    Tensor tmp = biases_;
    const auto ws = cast(weights_.dims_[1]);

    auto in_ = in.begin();
    auto out_ = tmp.begin();
    for (auto w = weights_.begin(); w < weights_.end(); w += ws)
        *(out_++) += std::inner_product(w, w + ws, in_, 0.f);

    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
