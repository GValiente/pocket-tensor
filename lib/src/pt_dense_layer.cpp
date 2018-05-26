/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_dense_layer.h"

#include "pt_multiply_add.h"
#include "pt_logger.h"

namespace pt
{

namespace
{
    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& weights, const Tensor& in, Tensor& out) noexcept
    {
        const auto& weightsDims = weights.getDims();
        auto wi = int(weightsDims[1]);
        auto inIt = in.begin();
        auto outIt = out.begin();
        MultiplyAddType multiplyAdd;

        for(auto w = weights.begin(), wl = weights.end(); w != wl; w += wi)
        {
            *(outIt++) += multiplyAdd(&*inIt, &*w, wi);
        }
    }
}

std::unique_ptr<DenseLayer> DenseLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(2, stream);

    if(! weights)
    {
        PT_LOG_ERROR << "Weights tensor parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    auto biases = Tensor::create(1, stream);

    if(! biases)
    {
        PT_LOG_ERROR << "Biases tensor parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<DenseLayer>();
    }

    return std::unique_ptr<DenseLayer>(new DenseLayer(std::move(*weights), std::move(*biases),
                                                      std::move(activation)));
}

bool DenseLayer::apply(const Config& config, Tensor&& in, Tensor& out) const
{
    const auto& iw = in.getDims();

    if(iw.size() != 1)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 1" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[0] != ww[1])
    {
        PT_LOG_ERROR << "Input tensor dims[0] must be the same as weights dims[1]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{_weights.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    _biases.copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && ww[1] % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(_weights, in, out);
    }
    else
    {
        multiplyAddImpl<VectorMultiplyAdd>(_weights, in, out);
    }

    _activation->apply(config, out);
    return true;
}

DenseLayer::DenseLayer(Tensor&& weights, Tensor&& biases,
                       std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
    _weights.addPadding();
    _biases.addPadding();
}

}
