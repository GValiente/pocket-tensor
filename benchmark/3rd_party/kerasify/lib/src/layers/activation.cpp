/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/activation.h"

namespace keras {
namespace layers {

bool Activation::load_layer(std::ifstream& file) noexcept
{
    unsigned activation = 0;
    check(read_uint(file, activation));

    switch (activation) {
    case Linear:
        activation_type_ = Linear;
        break;
    case Relu:
        activation_type_ = Relu;
        break;
    case Elu:
        activation_type_ = Elu;
        break;
    case SoftPlus:
        activation_type_ = SoftPlus;
        break;
    case SoftSign:
        activation_type_ = SoftSign;
        break;
    case HardSigmoid:
        activation_type_ = HardSigmoid;
        break;
    case Sigmoid:
        activation_type_ = Sigmoid;
        break;
    case Tanh:
        activation_type_ = Tanh;
        break;
    default:
        check(false);
    }
    return true;
}

bool Activation::apply(const Tensor& in, Tensor& out) const noexcept
{
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    switch (activation_type_) {
    case Linear:
        std::copy(in.begin(), in.end(), out.begin());
        break;
    case Relu:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x < 0.f)
                return 0.f;
            return x;
        });
        break;
    case Elu:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x < 0.f)
                return std::expm1(x);
            return x;
        });
        break;
    case SoftPlus:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return std::log1p(std::exp(x));
        });
        break;
    case SoftSign:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return x / (1.f + std::abs(x));
        });
        break;
    case HardSigmoid:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x <= -2.5f)
                return 0.f;
            if (x >= 2.5f)
                return 1.f;
            return (x * .2f) + .5f;
        });
        break;
    case Sigmoid:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            float z = std::exp(-std::abs(x));
            if (x < 0)
                return z / (1.f + z);
            return 1.f / (1.f + z);
        });
        break;
    case Tanh:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return std::tanh(x);
        });
        break;
    }
    return true;
}

} // namespace layers
} // namespace keras
