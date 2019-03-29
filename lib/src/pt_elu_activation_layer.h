/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_ELU_ACTIVATION_LAYER_H
#define PT_ELU_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class EluActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    EluActivationLayer() = default;

    void apply(Tensor& out) const final
    {
        for(FloatType& value : out)
        {
            if(value < 0)
            {
                value = std::expm1(value);
            }
        }
    }
};

}

#endif
