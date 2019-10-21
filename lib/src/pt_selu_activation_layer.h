/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_SELU_ACTIVATION_LAYER_H
#define PT_SELU_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class SeluActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    SeluActivationLayer() = default;

    void apply(Tensor& out) const final
    {
        constexpr auto alpha = FloatType(1.6732632423543772848170429916717);
        constexpr auto scale = FloatType(1.0507009873554804934193349852946);

        for(FloatType& value : out)
        {
            if(value < 0)
            {
                value = alpha * std::expm1(value);
            }

            value *= scale;
        }
    }
};

}

#endif
