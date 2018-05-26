/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_conv_1d_layer.h"

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
        const auto& ww = weights.getDims();
        const auto& ow = out.getDims();
        auto outInc = int(ow[1]);
        auto ws0 = int(ww[2] * ww[1]);
        auto ws1 = int(ww[2]);

        auto tx = int(ow[0]);

        auto inBegin = in.begin();
        auto outBegin = out.begin();
        auto wBegin = weights.begin();
        auto wEnd = weights.end();
        auto bBegin = biases.begin();
        MultiplyAddType multiplyAdd;

        for(int x = 0; x != tx; ++x)
        {
            auto inIt = inBegin + x * ws1;
            auto outIt = outBegin;
            auto bIt = bBegin;
            outBegin += outInc;

            for(auto w0 = wBegin; w0 != wEnd; w0 += ws0)
            {
                *outIt = *bIt + multiplyAdd(&*inIt, &*w0, ws0);
                ++outIt;
                ++bIt;
            }
        }
    }
}

std::unique_ptr<Conv1DLayer> Conv1DLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(3, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<Conv1DLayer>();
    }

    auto biases = Tensor::create(1, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return std::unique_ptr<Conv1DLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<Conv1DLayer>();
    }

    return std::unique_ptr<Conv1DLayer>(new Conv1DLayer(std::move(*weights), std::move(*biases),
                                                        std::move(activation)));
}

bool Conv1DLayer::apply(const Config& config, Tensor&& in, Tensor& out) const
{
    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[1] != ww[2])
    {
        PT_LOG_ERROR << "Input tensor dims[1] must be the same as weights dims[2]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{_weights.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    auto offset = ww[1] - 1;
    out.resizeWithPadding(iw[0] - offset, ww[0]);

    if(PT_LOOP_UNROLLING_ENABLE && (ww[2] * ww[1]) % (Tensor::VectorSize * 2) == 0)
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

Conv1DLayer::Conv1DLayer(Tensor&& weights, Tensor&& biases,
                         std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
    _weights.addPadding();
    _biases.addPadding();
}

}
