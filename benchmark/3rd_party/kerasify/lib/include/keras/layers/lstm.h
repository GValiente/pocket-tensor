/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {
namespace layers {

class LSTM final : public Layer {
public:
    bool load_layer(std::ifstream& file) noexcept override;
    bool apply(const Tensor& in, Tensor& out) const noexcept override;

private:
    bool step(const Tensor& x, Tensor& out, Tensor& ht_1, Tensor& ct_1) const
        noexcept;

    Tensor Wi_;
    Tensor Ui_;
    Tensor bi_;
    Tensor Wf_;
    Tensor Uf_;
    Tensor bf_;
    Tensor Wc_;
    Tensor Uc_;
    Tensor bc_;
    Tensor Wo_;
    Tensor Uo_;
    Tensor bo_;

    Activation inner_activation_;
    Activation activation_;
    bool return_sequences_{false};
};

} // namespace layers
} // namespace keras
