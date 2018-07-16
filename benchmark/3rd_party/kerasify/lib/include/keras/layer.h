/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/tensor.h"

namespace keras {

class Layer {
public:
    virtual ~Layer();
    virtual bool load_layer(std::ifstream& file) noexcept = 0;
    virtual bool apply(const Tensor& in, Tensor& out) const noexcept = 0;
};

} // namespace keras
