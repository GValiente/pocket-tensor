/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/elu.h"

namespace keras {
namespace layers {

bool ELU::load_layer(std::ifstream& file) noexcept
{
    check(read_float(file, alpha_));
    return true;
}

bool ELU::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() > 0);
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    std::transform(in.begin(), in.end(), out.begin(), [this](float x) {
        if (x >= 0.f)
            return x;
        return alpha_ * std::expm1(x);
    });
    return true;
}

} // namespace layers
} // namespace keras
