/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_batch_normalization_layer.h"

#include "pt_layer_data.h"
#include "pt_logger.h"

namespace pt
{

std::unique_ptr<BatchNormalizationLayer> BatchNormalizationLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(1, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return nullptr;
    }

    auto biases = Tensor::create(1, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return nullptr;
    }

    if(weights->getDims() != biases->getDims())
    {
        PT_LOG_ERROR << "Invalid biases tensor dims" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<BatchNormalizationLayer>(
                new BatchNormalizationLayer(std::move(*weights), std::move(*biases)));
}

bool BatchNormalizationLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;

    if(in.getDims() != _weights.getDims())
    {
        PT_LOG_ERROR << "Input and weights tensor dims are different" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ in.getDims() } << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{ _weights.getDims() } << ")" << std::endl;
        return false;
    }

    in.fma(_weights, _biases, layerData.out, layerData.dispatcher);
    return true;
}

BatchNormalizationLayer::BatchNormalizationLayer(Tensor&& weights, Tensor&& biases) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases))
{
}

}
