/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_REPEAT_VECTOR_LAYER_H
#define PT_REPEAT_VECTOR_LAYER_H

#include "pt_layer.h"

namespace pt
{

class RepeatVectorLayer : public Layer
{

public:
    static std::unique_ptr<RepeatVectorLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    int _n;

    explicit RepeatVectorLayer(int n) noexcept;
};

}

#endif
