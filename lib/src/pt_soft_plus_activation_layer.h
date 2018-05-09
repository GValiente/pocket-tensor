/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_SOFT_PLUS_ACTIVATION_LAYER_H
#define PT_SOFT_PLUS_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class SoftPlusActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    SoftPlusActivationLayer() = default;

    void apply(const Config&, Tensor& out) const final
    {
        for(FloatType& value : out)
        {
            value = std::log1p(std::exp(value));
        }
    }
};

}

#endif
