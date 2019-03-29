/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LEAKY_RELU_LAYER_H
#define PT_LEAKY_RELU_LAYER_H

#include "pt_libsimdpp.h"
#include "pt_layer.h"

namespace pt
{

class LeakyReluLayer : public Layer
{

public:
    static std::unique_ptr<LeakyReluLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    FloatType _alpha;

    explicit LeakyReluLayer(FloatType alpha) noexcept;
};

}

#endif
