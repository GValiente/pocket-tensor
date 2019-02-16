/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_GLOBAL_MAX_POOLING_2D_LAYER_H
#define PT_GLOBAL_MAX_POOLING_2D_LAYER_H

#include "pt_layer.h"

namespace pt
{

class GlobalMaxPooling2DLayer : public Layer
{

public:
    static std::unique_ptr<GlobalMaxPooling2DLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    GlobalMaxPooling2DLayer() = default;
};

}

#endif
