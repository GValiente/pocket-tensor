/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {
namespace layers {

class Conv2D final : public Layer {
public:
    bool load_layer(std::ifstream& file) noexcept override;
    bool apply(const Tensor& in, Tensor& out) const noexcept override;

private:
    Tensor weights_;
    Tensor biases_;
    Activation activation_;
};

} // namespace layers
} // namespace keras
