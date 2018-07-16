/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/conv1d.h"

namespace keras {
namespace layers {

bool Conv1D::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file, 3));
    check(biases_.load(file));
    check(activation_.load_layer(file));
    return true;
}

bool Conv1D::apply(const Tensor& in, Tensor& out) const noexcept
{
    // 'in' have shape (steps, features)
    // 'tmp' have shape (new_steps, outputs)
    // 'weights' have shape (outputs, kernel, features)
    check(in.dims_[1] == weights_.dims_[2]);

    auto& ww = weights_.dims_;

    size_t offset = ww[1] - 1;
    Tensor tmp{in.dims_[0] - offset, ww[0]};

    auto ts0 = cast(ww[0]);
    auto ws0 = cast(ww[2] * ww[1]);
    auto ws1 = cast(ww[2]);

    auto tx = cast(tmp.dims_[0]);

    auto b_ptr = biases_.begin();
    auto t_ptr = tmp.begin();
    auto i_ptr = in.begin();

    for (ptrdiff_t x = 0; x < tx; ++x) {
        auto b_ = b_ptr;
        auto i_ = i_ptr + x * ws1;
        auto t_ = t_ptr + x * ts0;
        for (auto w0 = weights_.end(); w0 < weights_.end(); w0 += ws0)
            *(t_++) = std::inner_product(w0, w0 + ws0, i_, *(b_++));
    }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
