/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LINEAR_ACTIVATION_LAYER_H
#define PT_LINEAR_ACTIVATION_LAYER_H

#include "pt_activation_layer.h"

namespace pt
{

class LinearActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    LinearActivationLayer() = default;

    void apply(Tensor&) const final
    {
    }
};

}

#endif
