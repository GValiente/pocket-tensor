/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_elu_layer.h"

#include "pt_parser.h"
#include "pt_layer_data.h"

namespace pt
{

std::unique_ptr<EluLayer> EluLayer::create(std::istream& stream)
{
    float alpha = 0;

    if(! Parser::parse(stream, alpha))
    {
        PT_LOG_ERROR << "Alpha parse failed" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<EluLayer>(new EluLayer(FloatType(alpha)));
}

bool EluLayer::apply(LayerData& layerData) const
{
    layerData.out = std::move(layerData.in);

    for(FloatType& value : layerData.out)
    {
        if(value < 0)
        {
            value = _alpha * std::expm1(value);
        }
    }

    return true;
}

EluLayer::EluLayer(FloatType alpha) noexcept :
    _alpha(alpha)
{
}

}
