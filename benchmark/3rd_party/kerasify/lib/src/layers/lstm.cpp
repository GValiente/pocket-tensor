/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/lstm.h"

namespace keras {
namespace layers {

bool LSTM::load_layer(std::ifstream& file) noexcept
{
    // Load Input Weights and Biases
    check(Wi_.load(file, 2));
    check(Ui_.load(file, 2));
    check(bi_.load(file, 2));

    // Load Forget Weights and Biases
    check(Wf_.load(file, 2));
    check(Uf_.load(file, 2));
    check(bf_.load(file, 2));

    // Load State Weights and Biases
    check(Wc_.load(file, 2));
    check(Uc_.load(file, 2));
    check(bc_.load(file, 2));

    // Load Output Weights and Biases
    check(Wo_.load(file, 2));
    check(Uo_.load(file, 2));
    check(bo_.load(file, 2));

    check(inner_activation_.load_layer(file));
    check(activation_.load_layer(file));

    unsigned return_sequences = 0;
    check(read_uint(file, return_sequences));
    return_sequences_ = static_cast<bool>(return_sequences);
    return true;
}

bool LSTM::apply(const Tensor& in, Tensor& out) const noexcept
{
    // Assume 'bo_' always keeps the output shape and we will always
    // receive one single sample.
    size_t out_dim = bo_.dims_[1];
    size_t steps = in.dims_[0];

    Tensor ht_1{1, out_dim};
    Tensor ct_1{1, out_dim};

    ht_1.fill(0.f);
    ct_1.fill(0.f);

    if (!return_sequences_) {
        for (size_t s = 0; s < steps; ++s)
            check(step(in.select(s), out, ht_1, ct_1));
        return true;
    }

    out.dims_ = {steps, out_dim};
    out.data_.reserve(steps * out_dim);

    Tensor last;
    for (size_t s = 0; s < steps; ++s) {
        check(step(in.select(s), last, ht_1, ct_1));
        out.data_.insert(out.end(), last.begin(), last.end());
    }
    return true;
}

bool LSTM::step(const Tensor& x, Tensor& out, Tensor& ht_1, Tensor& ct_1) const
    noexcept
{
    Tensor xi = x.dot(Wi_) + bi_;
    Tensor xf = x.dot(Wf_) + bf_;
    Tensor xc = x.dot(Wc_) + bc_;
    Tensor xo = x.dot(Wo_) + bo_;

    Tensor i_ = xi + ht_1.dot(Ui_);
    Tensor f_ = xf + ht_1.dot(Uf_);
    Tensor c_ = xc + ht_1.dot(Uc_);
    Tensor o_ = xo + ht_1.dot(Uo_);

    Tensor i, f, cc, o;

    check(inner_activation_.apply(i_, i));
    check(inner_activation_.apply(f_, f));
    check(activation_.apply(c_, cc));
    check(inner_activation_.apply(o_, o));

    auto m1 = f.multiply(ct_1);
    auto m2 = i.multiply(cc);
    ct_1 = m1 + m2;

    check(activation_.apply(ct_1, cc));

    out = ht_1 = o.multiply(cc);
    return true;
}

} // namespace layers
} // namespace keras
