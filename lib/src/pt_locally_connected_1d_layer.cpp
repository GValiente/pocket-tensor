/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_locally_connected_1d_layer.h"

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
        const auto& iw = in.getDims();
        auto is0 = int(iw[1]);
        auto ts0 = int(ww[1]);
        auto ws0 = int(ww[2] * ww[1]);
        auto ws1 = int(ww[2]);

        auto inIt = in.begin();
        auto outIt = out.begin();
        auto bIt = biases.begin();
        MultiplyAddType multiplyAdd;

        for(auto wIt = weights.begin(), wEndIt = weights.end(); wIt != wEndIt;
                wIt += ws0, bIt += ts0, outIt += ts0, inIt += is0)
        {
            auto outIt2 = outIt;
            auto bIt2 = bIt;

            for(auto w0 = wIt; w0 != wIt + ws0; w0 += ws1)
            {
                *outIt2 = *bIt2 + multiplyAdd(&*inIt, &*w0, ws1);
                ++outIt2;
                ++bIt2;
            }
        }
    }
}

std::unique_ptr<LocallyConnected1DLayer> LocallyConnected1DLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(3, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<LocallyConnected1DLayer>();
    }

    auto biases = Tensor::create(2, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return std::unique_ptr<LocallyConnected1DLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<LocallyConnected1DLayer>();
    }

    return std::unique_ptr<LocallyConnected1DLayer>(
                new LocallyConnected1DLayer(std::move(*weights), std::move(*biases),
                                            std::move(activation)));
}

bool LocallyConnected1DLayer::apply(const Config& config, Tensor&& in, Tensor& out) const
{
    in.removePadding();

    if(out.isValid())
    {
        out.removePadding(false);
    }

    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{iw} << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();
    auto offset = (ww[2] / iw[1]) - 1;

    if(iw[0] != ww[0] + offset)
    {
        PT_LOG_ERROR << "Input tensor dims[0] must be the same as weights dims[0] + offset" <<
                            " (input dims: " << VectorPrinter<std::size_t>{iw} << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{ww} << ")" <<
                            " (offset: " << offset << ")" << std::endl;
        return false;
    }

    auto ws1 = ww[2];
    out.resize(ww[0], ww[1]);

    if(PT_LOOP_UNROLLING_ENABLE && ws1 % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(_weights, _biases, in, out);
    }
    else if(ws1 % Tensor::VectorSize == 0)
    {
        multiplyAddImpl<VectorMultiplyAdd>(_weights, _biases, in, out);
    }
    else
    {
        multiplyAddImpl<ScalarMultiplyAdd>(_weights, _biases, in, out);
    }

    out.addPadding();
    _activation->apply(config, out);
    return true;
}

LocallyConnected1DLayer::LocallyConnected1DLayer(Tensor&& weights, Tensor&& biases,
                                                 std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
    _weights.removePadding();
    _biases.removePadding();
}

}
