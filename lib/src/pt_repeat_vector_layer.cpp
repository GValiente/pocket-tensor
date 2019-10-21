/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_repeat_vector_layer.h"

#include "pt_parser.h"
#include "pt_layer_data.h"

namespace pt
{

std::unique_ptr<RepeatVectorLayer> RepeatVectorLayer::create(std::istream& stream)
{
    int n = 0;

    if(! Parser::parse(stream, n))
    {
        PT_LOG_ERROR << "n parse failed" << std::endl;
        return nullptr;
    }

    if(n <= 0)
    {
        PT_LOG_ERROR << "invalid n: " << n << std::endl;
        return nullptr;
    }

    return std::unique_ptr<RepeatVectorLayer>(new RepeatVectorLayer(n));
}

bool RepeatVectorLayer::apply(LayerData& layerData) const
{
    layerData.in.repeat(_n, 1, layerData.out);
    return true;
}

RepeatVectorLayer::RepeatVectorLayer(int n) noexcept :
    _n(n)
{
}

}
