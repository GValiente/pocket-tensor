/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_INPUT_LAYER_H
#define PT_INPUT_LAYER_H

#include "pt_layer.h"
#include "pt_layer_data.h"

namespace pt
{

class InputLayer : public Layer
{

public:
    static std::unique_ptr<InputLayer> create(std::istream&)
    {
        return std::unique_ptr<InputLayer>(new InputLayer());
    }

    bool apply(LayerData& layerData) const final
    {
        layerData.out = std::move(layerData.in);
        return true;
    }

protected:
    InputLayer() = default;
};

}

#endif
