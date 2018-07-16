/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/flatten.h"

namespace keras {
namespace layers {

bool Flatten::load_layer(std::ifstream&) noexcept { return true; }

bool Flatten::apply(const Tensor& in, Tensor& out) const noexcept
{
    out = in;
    out.flatten();
    return true;
}

} // namespace layers
} // namespace keras
