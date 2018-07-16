/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/embedding.h"

namespace keras {
namespace layers {

bool Embedding::load_layer(std::ifstream& file) noexcept
{
    check(weights_.load(file, 2));
    return true;
}

bool Embedding::apply(const Tensor& in, Tensor& out) const noexcept
{
    size_t out_i = in.dims_[0];
    size_t out_j = weights_.dims_[1];

    out.data_.reserve(out_i * out_j);
    out.dims_ = {out_i, out_j};

    for (const auto& it : in.data_) {
        auto first = weights_.begin() + cast(it * out_j);
        auto last = weights_.begin() + cast(it * out_j + out_j);
        out.data_.insert(out.end(), first, last);
    }
    return true;
}
} // namespace layers
} // namespace keras
