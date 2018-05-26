/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_embedding_layer.h"

#include "pt_logger.h"

namespace pt
{

std::unique_ptr<EmbeddingLayer> EmbeddingLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(2, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<EmbeddingLayer>();
    }

    return std::unique_ptr<EmbeddingLayer>(new EmbeddingLayer(std::move(*weights)));
}

bool EmbeddingLayer::apply(const Config&, Tensor&& in, Tensor& out) const
{
    const auto iw = in.getUnpaddedDims();

    if(iw.size() != 1)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 1" <<
                            " (input dims: " << VectorPrinter<std::size_t>{iw} << ")" << std::endl;
        return false;
    }

    out.resizeWithPadding(iw[0], _weights.getUnpaddedDims()[1]);

    auto outIt = out.begin();
    auto wBegin = _weights.begin();
    auto inc = _weights.getDims()[1];

    for(auto inIt = in.begin(), inEnd = in.begin() + long(iw[0]); inIt != inEnd; ++inIt)
    {
        auto wIt = wBegin + int(*inIt * inc);
        auto wPtr = &*wIt;
        auto oPtr = &*outIt;
        outIt += long(inc);

        for(int index = 0; index != int(inc); index += Tensor::VectorSize)
        {
            Tensor::Vector wv = simdpp::load(wPtr + index);
            simdpp::store(oPtr + index, wv);
        }
    }

    return true;
}

EmbeddingLayer::EmbeddingLayer(Tensor&& weights) noexcept :
    _weights(std::move(weights))
{
    _weights.addPadding();
}

}
