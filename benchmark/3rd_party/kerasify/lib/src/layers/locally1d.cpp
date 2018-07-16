/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/locally1d.h"

namespace keras {
namespace layers {

bool LocallyConnected1D::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file, 3));
    check(biases_.load(file, 2));
    check(activation_.load_layer(file));
    return true;
}

bool LocallyConnected1D::apply(const Tensor& in, Tensor& out) const noexcept
{
    // 'in' have shape (steps, features)
    // 'tmp' have shape (new_steps, outputs)
    // 'weights' have shape (new_steps, outputs, kernel*features)
    // 'biases' have shape (new_steps, outputs)
    auto& ww = weights_.dims_;

    size_t ksize = ww[2] / in.dims_[1];
    size_t offset = ksize - 1;
    check(in.dims_[0] - offset == ww[0]);

    Tensor tmp{ww[0], ww[1]};

    auto is0 = cast(in.dims_[1]);
    auto ts0 = cast(ww[1]);
    auto ws0 = cast(ww[2] * ww[1]);
    auto ws1 = cast(ww[2]);

    auto b_ptr = biases_.begin();
    auto t_ptr = tmp.begin();
    auto i_ptr = in.begin();

    for (auto w_ = weights_.begin(); w_ < weights_.end();
         w_ += ws0, b_ptr += ts0, t_ptr += ts0, i_ptr += is0) {
        auto b_ = b_ptr;
        auto t_ = t_ptr;
        auto i_ = i_ptr;
        for (auto w0 = w_; w0 < w_ + ws0; w0 += ws1)
            *(t_++) = std::inner_product(w0, w0 + ws1, i_, *(b_++));
    }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
