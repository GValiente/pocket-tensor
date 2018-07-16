/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/conv2d.h"

namespace keras {
namespace layers {

bool Conv2D::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file, 4));
    check(biases_.load(file));
    check(activation_.load_layer(file));
    return true;
}

bool Conv2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_[2] == weights_.dims_[3]);

    auto& ww = weights_.dims_;

    size_t offset_y = ww[1] - 1;
    size_t offset_x = ww[2] - 1;
    Tensor tmp{in.dims_[0] - offset_y, in.dims_[1] - offset_x, ww[0]};

    auto ts0 = cast(ww[0] * tmp.dims_[1]);
    auto ts1 = cast(ww[0]);
    auto ws_ = cast(ww[0] * ww[1] * ww[2] * ww[3]);
    auto ws0 = cast(ww[1] * ww[2] * ww[3]);
    auto ws1 = cast(ww[2] * ww[3]);
    auto ws2 = cast(ww[3]);
    auto is0 = cast(ww[3] * in.dims_[1]);

    auto ty = cast(tmp.dims_[0]);
    auto tx = cast(tmp.dims_[1]);

    auto w_ptr = weights_.begin();
    auto b_ptr = biases_.begin();
    auto t_ptr = tmp.begin();
    auto i_ptr = in.begin();

    for (ptrdiff_t y = 0; y < ty; ++y)
        for (ptrdiff_t x = 0; x < tx; ++x) {
            auto b_ = b_ptr;
            auto i_ = i_ptr + y * is0 + x * ws2;
            auto t_ = t_ptr + y * ts0 + x * ts1;
            for (auto w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0, ++t_) {
                *t_ = *(b_++); // init with bias
                auto i0 = i_;
                for (auto w1 = w0; w1 < w0 + ws0; w1 += ws1, i0 += is0)
                    *t_ += std::inner_product(w1, w1 + ws1, i0, 0.f);
            }
        }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
