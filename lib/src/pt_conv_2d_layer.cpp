/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_conv_2d_layer.h"

#include "pt_multiply_add.h"
#include "pt_logger.h"

namespace pt
{

namespace
{
    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& weights, const Tensor& biases, const Tensor& in,
                         Tensor& out) noexcept
    {
        const auto& iw = in.getDims();
        const auto& ww = weights.getDims();
        const auto& ow = out.getDims();
        auto outInc = int(ow[2]);
        auto ws = int(ww[0] * ww[1] * ww[2] * ww[3]);
        auto ws0 = int(ww[1] * ww[2] * ww[3]);
        auto ws1 = int(ww[2] * ww[3]);
        auto ws2 = int(ww[3]);
        auto is0 = int(ww[3] * iw[1]);

        auto ty = int(ow[0]);
        auto tx = int(ow[1]);

        auto inBegin = in.getData().data();
        auto outBegin = const_cast<Tensor::Type*>(out.getData().data());
        auto wBegin = weights.getData().data();
        auto bBegin = biases.getData().data();
        MultiplyAddType multiplyAdd;

        for(int y = 0; y != ty; ++y)
        {
            for(int x = 0; x != tx; ++x)
            {
                auto inIt = inBegin + y * is0 + x * ws2;
                auto outIt = outBegin;
                auto bIt = bBegin;
                outBegin += outInc;

                for(auto w0 = wBegin; w0 != wBegin + ws; w0 += ws0, ++outIt)
                {
                    auto i0 = inIt;
                    *outIt = *bIt;
                    ++bIt;

                    for(auto w1 = w0; w1 != w0 + ws0; w1 += ws1, i0 += is0)
                    {
                        *outIt += multiplyAdd(&*i0, &*w1, ws1);
                    }
                }
            }
        }
    }
}

std::unique_ptr<Conv2DLayer> Conv2DLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(4, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<Conv2DLayer>();
    }

    auto biases = Tensor::create(1, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return std::unique_ptr<Conv2DLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<Conv2DLayer>();
    }

    return std::unique_ptr<Conv2DLayer>(new Conv2DLayer(std::move(*weights), std::move(*biases),
                                                        std::move(activation)));
}

bool Conv2DLayer::apply(const Config& config, Tensor&& in, Tensor& out) const
{
    const auto& iw = in.getDims();

    if(iw.size() != 3)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 3" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[2] != ww[3])
    {
        PT_LOG_ERROR << "Input tensor dims[2] must be the same as weights dims[3]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{_weights.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    auto offsetY = ww[1] - 1;
    auto offsetX = ww[2] - 1;
    out.resizeWithPadding(iw[0] - offsetY, iw[1] - offsetX, ww[0]);

    if(PT_LOOP_UNROLLING_ENABLE && (ww[2] * ww[3]) % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(_weights, _biases, in, out);
    }
    else
    {
        multiplyAddImpl<VectorMultiplyAdd>(_weights, _biases, in, out);
    }

    _activation->apply(config, out);
    return true;
}

Conv2DLayer::Conv2DLayer(Tensor&& weights, Tensor&& biases,
                         std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
    _weights.addPadding();
    _biases.addPadding();
}

}
