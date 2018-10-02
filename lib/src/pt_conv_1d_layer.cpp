/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_conv_1d_layer.h"

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

                const auto& ww = weights->getDims();
                const auto& ow = out.getDims();
                auto outInc = int(ow[1]);
                auto wInc = int(ww[2] * ww[1]);
                auto wInc2 = int(ww[2]);

                auto tx = int(ow[0]);

                auto inBegin = in.begin();
                auto outBegin = out.begin();
                auto wBegin = weights->begin();
                auto wEnd = weights->end();
                auto bBegin = biases->begin();
                MultiplyAddType multiplyAdd;

                int its = tx;
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

                for(int x = taskBegin; x != taskEnd; ++x)
                {
                    auto inIt = inBegin + x * wInc2;
                    auto outIt = outBegin + x * outInc;
                    auto bIt = bBegin;

                    for(auto wIt = wBegin; wIt != wEnd; wIt += wInc)
                    {
                        *outIt = *bIt + multiplyAdd(&*inIt, &*wIt, wInc);
                        ++outIt;
                        ++bIt;
                    }
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks{};
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

bool Conv1DLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    const auto& ww = _weights.getDims();

    if(iw[1] != ww[2])
    {
        PT_LOG_ERROR << "Input tensor dims[1] must be the same as weights dims[2]" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" <<
                            " (weights dims: " << VectorPrinter<std::size_t>{ ww } << ")" << std::endl;
        return false;
    }

    auto offset = ww[1] - 1;
    Tensor& out = layerData.out;
    out.resize(iw[0] - offset, ww[0]);

    auto threads = int(layerData.dispatcher.threads());
    auto threadSize = int(ww[2] * ww[1]) / threads;

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

Conv1DLayer::Conv1DLayer(Tensor&& weights, Tensor&& biases,
                         std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
}

}
