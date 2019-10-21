/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LOCALLY_CONNECTED_1D_LAYER_H
#define PT_LOCALLY_CONNECTED_1D_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class LocallyConnected1DLayer : public Layer
{

public:
    static std::unique_ptr<LocallyConnected1DLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    Tensor _weights;
    Tensor _biases;
    std::unique_ptr<ActivationLayer> _activation;

    LocallyConnected1DLayer(Tensor&& weights, Tensor&& biases,
			    std::unique_ptr<ActivationLayer>&& activation) noexcept;
};

}

#endif
