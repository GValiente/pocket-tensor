/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_locally_connected_1d_layer.h"

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
                const auto& iw = in.getDims();
                auto inInc = int(iw[1]);
                auto bOutInc = int(ww[1]);
                auto wInc = int(ww[2] * ww[1]);
                auto wInc2 = int(ww[2]);

                auto inIt = in.begin();
                auto outIt = out.begin();
                auto bIt = biases->begin();
                MultiplyAddType multiplyAdd;

                auto weightsBegin = weights->begin();
                int its = int(weights->end() - weightsBegin) / wInc;
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

                inIt += taskIts * taskId * inInc;
                outIt += taskIts * taskId * bOutInc;
                bIt += taskIts * taskId * bOutInc;

                for(auto wIt = weightsBegin + (taskBegin * wInc), wEnd = weightsBegin + (taskEnd * wInc);
                    wIt != wEnd; wIt += wInc)
                {
                    auto outIt2 = outIt;
                    auto bIt2 = bIt;

                    for(auto wIt2 = wIt; wIt2 != wIt + wInc; wIt2 += wInc2)
                    {
                        *outIt2 = *bIt2 + multiplyAdd(&*inIt, &*wIt2, wInc2);
                        ++outIt2;
                        ++bIt2;
                    }

                    inIt += inInc;
                    outIt += bOutInc;
                    bIt += bOutInc;
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        Dispatcher& dispatcher = layerData.dispatcher;
        int threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &weights, &biases, &layerData, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
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

bool LocallyConnected1DLayer::apply(LayerData& layerData) const
{
    Tensor& in = layerData.in;
    Tensor& out = layerData.out;
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

    out.resize(ww[0], ww[1]);

    int threads = int(layerData.dispatcher.threads());
    int threadSize = int(ww[2]) / threads;

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

LocallyConnected1DLayer::LocallyConnected1DLayer(Tensor&& weights, Tensor&& biases,
                                                 std::unique_ptr<ActivationLayer>&& activation) noexcept :
    _weights(std::move(weights)),
    _biases(std::move(biases)),
    _activation(std::move(activation))
{
}

}
