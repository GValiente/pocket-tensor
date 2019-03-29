/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_leaky_relu_layer.h"

#include "pt_parser.h"
#include "pt_layer_data.h"

namespace pt
{

std::unique_ptr<LeakyReluLayer> LeakyReluLayer::create(std::istream& stream)
{
    float alpha = 0;

    if(! Parser::parse(stream, alpha))
    {
        PT_LOG_ERROR << "Alpha parse failed" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<LeakyReluLayer>(new LeakyReluLayer(FloatType(alpha)));
}

bool LeakyReluLayer::apply(LayerData& layerData) const
{
    layerData.out = std::move(layerData.in);

    for(FloatType& value : layerData.out)
    {
        if(value < 0)
        {
            value *= _alpha;
        }
    }

    return true;
}

LeakyReluLayer::LeakyReluLayer(FloatType alpha) noexcept :
    _alpha(alpha)
{
}

}
