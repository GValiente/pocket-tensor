/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/maxpooling2d.h"

namespace keras {
namespace layers {

bool MaxPooling2D::load_layer(std::ifstream& file) noexcept
{
    check(read_uint(file, pool_size_y_));
    check(read_uint(file, pool_size_x_));
    return true;
}

bool MaxPooling2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() == 3);

    const auto& iw = in.dims_;

    out.resize(iw[0] / pool_size_y_, iw[1] / pool_size_x_, iw[2]);
    out.fill(-std::numeric_limits<float>::infinity());

    auto is0p = cast(iw[2] * iw[1] * pool_size_y_);
    auto is0 = cast(iw[2] * iw[1]);
    auto is1p = cast(iw[2] * pool_size_x_);
    auto is1 = cast(iw[2]);
    auto os_ = cast(iw[2] * out.dims_[1] * out.dims_[0]);
    auto os0 = cast(iw[2] * out.dims_[1]);

    auto o_ptr = out.begin();
    auto i_ptr = in.begin();
    for (auto o0 = o_ptr; o0 < o_ptr + os_; o0 += os0, i_ptr += is0p) {
        auto i_ = i_ptr;
        for (auto o1 = o0; o1 < o0 + os0; o1 += is1, i_ += is1p)
            for (auto i0 = i_; i0 < i_ + is0p; i0 += is0)
                for (auto i1 = i0; i1 < i0 + is1p; i1 += is1)
                    std::transform(i1, i1 + is1, o1, o1, [](float x, float y) {
                        return std::max(x, y);
                    });
    }
    return true;
}

} // namespace layers
} // namespace keras
