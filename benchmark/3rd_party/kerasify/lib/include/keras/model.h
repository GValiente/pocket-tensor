/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"
#include <memory>

namespace keras {

class Model {
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    enum layer_type {
        Dense = 1,
        Conv1D = 2,
        Conv2D = 3,
        LocallyConnected1D = 4,
        LocallyConnected2D = 5,
        Flatten = 6,
        ELU = 7,
        Activation = 8,
        MaxPooling2D = 9,
        LSTM = 10,
        Embedding = 11,
        BatchNormalization = 12,
    };

    virtual ~Model() = default;
    virtual bool load_model(const std::string& filename) noexcept;
    virtual bool apply(const Tensor& in, Tensor& out) const noexcept;
};

} // namespace keras
