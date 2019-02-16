/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_conv_2d_layer.h"

#include <array>
#include "pt_dispatcher.h"
#include "pt_layer_data.h"
#include "pt_multiply_add.h"
#include "pt_logger.h"

namespace pt
{

namespace
{
    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& weights, const Tensor& biases, LayerData& layerData)
    {
        struct Task
        {
            const Tensor* weights;
            const Tensor* biases;
            LayerData* layerData;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                const Tensor& in = layerData->in;
                Tensor& out = layerData->out;

                const auto& iw = in.getDims();
                const auto& ww = weights->getDims();
                const auto& ow = out.getDims();
                auto outInc = int(ow[2]);
                auto wSize = int(ww[0] * ww[1] * ww[2] * ww[3]);
                auto wInc = int(ww[1] * ww[2] * ww[3]);
                auto wInc2 = int(ww[2] * ww[3]);

                auto tx = int(ow[1]);
                auto ty = int(ow[0]);
                auto inIncX = int(ww[3]);
                auto inIncY = int(ww[3] * iw[1]);

                auto inBegin = in.getData().data();
                auto outBegin = const_cast<Tensor::Type*>(out.getData().data());
                auto wBegin = weights->getData().data();
                auto bBegin = biases->getData().data();
                MultiplyAddType multiplyAdd;

                int its = ty;
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                for(int y = taskBegin; y != taskEnd; ++y)
                {
                    for(int x = 0; x != tx; ++x)
                    {
                        auto inIt = inBegin + y * inIncY + x * inIncX;
                        auto outIt = outBegin + y * tx * outInc + x * outInc;
                        auto bIt = bBegin;

                        for(auto wIt = wBegin, wEnd = wBegin + wSize; wIt != wEnd; wIt += wInc)
                        {
                            auto inIt2 = inIt;
                            *outIt = *bIt;

                            for(auto wIt2 = wIt, wEnd2 = wIt + wInc; wIt2 != wEnd2; wIt2 += wInc2)
                            {
                                *outIt += multiplyAdd(&*inIt2, &*wIt2, wInc2);
                                inIt2 += inIncY;
                            }

                            ++outIt;
                            ++bIt;
                        }
                    }
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        Dispatcher& dispatcher = layerData.dispatcher;
        auto threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &weights, &biases, &layerData, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }
}

std::unique_ptr<Conv2DLayer> Conv2DLayer::create(std::istream& stream)
{
    auto weights = Tensor::create(4, stream);

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

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<Conv2DLayer>(new Conv2DLayer(std::move(*weights), std::move(*biases),
                                                        std::move(activation)));
}

bool Conv2DLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 3)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 3" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[2] != ww[3])
    {
        PT_LOG_ERROR << "Input tensor dims[2] must be the same as weights dims[3]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{ ww } << ")" << std::endl;
        return false;
    }

    auto offsetY = ww[1] - 1;
    auto offsetX = ww[2] - 1;
    Tensor& out = layerData.out;
    out.resize(iw[0] - offsetY, iw[1] - offsetX, ww[0]);

    auto threads = int(layerData.dispatcher.threads());
    auto threadSize = int(ww[2] * ww[3]) / threads;

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(_weights, _biases, layerData);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        multiplyAddImpl<VectorMultiplyAdd>(_weights, _biases, layerData);
    }
    else
    {
        multiplyAddImpl<ScalarMultiplyAdd>(_weights, _biases, layerData);
    }

    _activation->apply(out);
    return true;
}

Conv2DLayer::Conv2DLayer(Tensor&& weights, Tensor&& biases,
                         std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
}

}
