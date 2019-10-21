/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_SIGMOID_ACTIVATION_LAYER_H
#define PT_SIGMOID_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class SigmoidActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    SigmoidActivationLayer() = default;

    void apply(Tensor& out) const final
    {
        for(FloatType& value : out)
        {
            FloatType z = std::exp(-std::abs(value));

            if(value < 0)
            {
                value = z / (FloatType(1) + z);
            }
            else
            {
                value = FloatType(1) / (FloatType(1) + z);
            }
        }
    }
};

}

#endif
