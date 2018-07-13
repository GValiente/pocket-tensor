/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_BATCH_NORMALIZATION_LAYER_H
#define PT_BATCH_NORMALIZATION_LAYER_H

#include "pt_layer.h"
#include "pt_tensor.h"

namespace pt
{

class BatchNormalizationLayer : public Layer
{

public:
    static std::unique_ptr<BatchNormalizationLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    Tensor _weights;
    Tensor _biases;

    BatchNormalizationLayer(Tensor&& weights, Tensor&& biases) noexcept;
};

}

#endif
