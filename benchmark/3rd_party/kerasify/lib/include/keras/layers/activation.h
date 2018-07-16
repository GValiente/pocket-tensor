/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Activation final : public Layer {
public:
    enum activation_type {
        Linear = 1,
        Relu = 2,
        Elu = 3,
        SoftPlus = 4,
        SoftSign = 5,
        Sigmoid = 6,
        Tanh = 7,
        HardSigmoid = 8
    };

    bool load_layer(std::ifstream& file) noexcept override;
    bool apply(const Tensor& in, Tensor& out) const noexcept override;

private:
    activation_type activation_type_{Linear};
};

} // namespace layers
} // namespace keras
