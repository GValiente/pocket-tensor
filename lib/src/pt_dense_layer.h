/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_DENSE_LAYER_H
#define PT_DENSE_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class DenseLayer : public Layer
{

public:
    static std::unique_ptr<DenseLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    Tensor _weights;
    Tensor _biases;
    std::unique_ptr<ActivationLayer> _activation;

    DenseLayer(Tensor&& weights, Tensor&& biases, std::unique_ptr<ActivationLayer>&& activation) noexcept;
};

}

#endif
