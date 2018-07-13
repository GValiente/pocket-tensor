/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_HARD_SIGMOID_ACTIVATION_LAYER_H
#define PT_HARD_SIGMOID_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class HardSigmoidActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    HardSigmoidActivationLayer() = default;

    void apply(Tensor& out) const final
    {
        for(FloatType& value : out)
        {
            if(value <= -FloatType(2.5))
            {
                value = 0;
            }
            else if(value >= FloatType(2.5))
            {
                value = 1;
            }
            else
            {
                value = (value * FloatType(0.2)) + FloatType(0.5);
            }
        }
    }
};

}

#endif
