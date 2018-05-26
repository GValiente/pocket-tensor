/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_max_pooling_2d_layer.h"

#include "pt_tensor.h"
#include "pt_parser.h"

namespace pt
{

std::unique_ptr<MaxPooling2DLayer> MaxPooling2DLayer::create(std::istream& stream)
{
    unsigned int poolSizeY = 0;

    if(! Parser::parse(stream, poolSizeY))
    {
        PT_LOG_ERROR << "Pool size Y parse failed" << std::endl;
        return std::unique_ptr<MaxPooling2DLayer>();
    }

    unsigned int poolSizeX = 0;

    if(! Parser::parse(stream, poolSizeX))
    {
        PT_LOG_ERROR << "Pool size X parse failed" << std::endl;
        return std::unique_ptr<MaxPooling2DLayer>();
    }

    return std::unique_ptr<MaxPooling2DLayer>(new MaxPooling2DLayer(int(poolSizeY), int(poolSizeX)));
}

bool MaxPooling2DLayer::apply(const Config&, Tensor&& in, Tensor& out) const
{
    const auto& iw = in.getDims();

    if(iw.size() != 3)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 3" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    out.resizeWithPadding(iw[0] / std::size_t(_poolSizeY), iw[1] / std::size_t(_poolSizeX),
            in.getUnpaddedDims()[2]);
    out.fill(-std::numeric_limits<Tensor::Type>::infinity());

    const auto& ow = out.getDims();
    auto is0 = int(iw[2] * iw[1]);
    auto is0p = is0 * _poolSizeY;
    auto is1 = int(iw[2]);
    auto is1p = is1 * _poolSizeX;
    auto os0 = int(iw[2] * ow[1]);
    auto os = os0 * int(ow[0]);

    auto inIt = in.getData().data();
    auto outIt = const_cast<Tensor::Type*>(out.getData().data());

    for(auto o0 = outIt; o0 != outIt + os; o0 += os0, inIt += is0p)
    {
        auto inIt2 = inIt;

        for(auto o1 = o0; o1 != o0 + os0; o1 += is1, inIt2 += is1p)
        {
            for(auto i0 = inIt2; i0 != inIt2 + is0p; i0 += is0)
            {
                for(auto i1 = i0; i1 != i0 + is1p; i1 += is1)
                {
                    for(int index = 0; index != is1; index += Tensor::VectorSize)
                    {
                        Tensor::Vector iv = simdpp::load(&(i1[index]));
                        Tensor::Vector ov = simdpp::load(&(o1[index]));
                        simdpp::store(&(o1[index]), simdpp::max(iv, ov));
                    }
                }
            }
        }
    }

    return true;
}

}
