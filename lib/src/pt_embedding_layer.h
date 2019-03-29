/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_EMBEDDING_LAYER_H
#define PT_EMBEDDING_LAYER_H

#include "pt_tensor.h"
#include "pt_layer.h"

namespace pt
{

class EmbeddingLayer : public Layer
{

public:
    static std::unique_ptr<EmbeddingLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    Tensor _weights;

    explicit EmbeddingLayer(Tensor&& weights) noexcept;
};

}

#endif
