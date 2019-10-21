/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_global_max_pooling_2d_layer.h"

#include <array>
#include "pt_parser.h"
#include "pt_dispatcher.h"
#include "pt_layer_data.h"
#include "pt_max.h"

namespace pt
{

namespace
{
    void maxImpl(LayerData& layerData)
    {
        struct Task
        {
            LayerData* layerData;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                const Tensor& in = layerData->in;
                Tensor& out = layerData->out;

                const auto& iw = in.getDims();
                auto its = out.getSize();
                auto taskIts = its / std::size_t(threads);
                auto taskBegin = taskIts * std::size_t(taskId);
                std::size_t taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto xl = iw[0];
                auto yl = iw[1];

                for(std::size_t z = taskBegin; z != taskEnd; ++z)
                {
                    Tensor::Type val = std::numeric_limits<Tensor::Type>::lowest();

                    for(std::size_t x = 0; x != xl; ++x)
                    {
                        for(std::size_t y = 0; y != yl; ++y)
                        {
                            val = std::max(val, in(x, y, z));
                        }
                    }

                    out(z) = val;
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        Dispatcher& dispatcher = layerData.dispatcher;
        auto threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &layerData, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }
}

std::unique_ptr<GlobalMaxPooling2DLayer> GlobalMaxPooling2DLayer::create(std::istream&)
{
    return std::unique_ptr<GlobalMaxPooling2DLayer>(new GlobalMaxPooling2DLayer());
}

bool GlobalMaxPooling2DLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 3)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 3" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    Tensor& out = layerData.out;
    out.resize(iw[2]);
    maxImpl(layerData);
    return true;
}

}
