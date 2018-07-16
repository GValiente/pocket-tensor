/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class ELU final : public Layer {
public:
    bool load_layer(std::ifstream& file) noexcept override;
    bool apply(const Tensor& in, Tensor& out) const noexcept override;

private:
    float alpha_{1.f};
};

} // namespace layers
} // namespace keras
