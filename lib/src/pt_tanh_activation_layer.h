/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_TANH_ACTIVATION_LAYER_H
#define PT_TANH_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class TanhActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    TanhActivationLayer() = default;

    void apply(const Config&, Tensor& out) const final
    {
        for(FloatType& value : out)
        {
            value = std::tanh(value);
        }
    }
};

}

#endif
