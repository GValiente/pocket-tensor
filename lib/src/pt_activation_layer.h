/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_ACTIVATION_LAYER_H
#define PT_ACTIVATION_LAYER_H

#include "pt_layer.h"

namespace pt
{

class ActivationLayer : public Layer
{

public:
    static std::unique_ptr<ActivationLayer> create(std::istream& stream);

    virtual void apply(const Config& config, Tensor& out) const = 0;

    bool apply(const Config& config, Tensor&& in, Tensor& out) const final;

protected:
    ActivationLayer() = default;
};

}

#endif
