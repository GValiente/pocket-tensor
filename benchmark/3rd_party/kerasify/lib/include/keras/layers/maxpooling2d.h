/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class MaxPooling2D final : public Layer {
public:
    bool load_layer(std::ifstream& file) noexcept override;
    bool apply(const Tensor& in, Tensor& out) const noexcept override;

private:
    unsigned pool_size_y_{0};
    unsigned pool_size_x_{0};
};

} // namespace layers
} // namespace keras
