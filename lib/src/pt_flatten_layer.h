/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_FLATTEN_LAYER_H
#define PT_FLATTEN_LAYER_H

#include "pt_layer.h"
#include "pt_layer_data.h"

namespace pt
{

class FlattenLayer : public Layer
{

public:
    FlattenLayer() = default;

    bool apply(LayerData& layerData) const final
    {
        layerData.out = std::move(layerData.in);
        layerData.out.flatten();
        return true;
    }
};

}

#endif
