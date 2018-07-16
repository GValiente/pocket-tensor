#include "pt_bench.h"

#include <chrono>
#include <string>
#include <iostream>
#include "pt_model.h"
#include "pt_tensor.h"
#include "pt_dispatcher.h"

namespace
{
    void testPocketTensorModel(const char* modelFileName, const pt::Tensor& in, int its)
    {
        auto modelPath = std::string(PT_BENCHMARK_MODELS_FOLDER) + "/pocket-tensor/" + modelFileName + ".model";
        auto model = pt::Model::create(modelPath);

        if(! model)
        {
            std::cerr << "pocket-tensor model load failed: " << modelPath << std::endl;
            return;
        }

        pt::Tensor out;
        pt::Dispatcher dispatcher;
        std::int64_t minElapsedMcs = std::numeric_limits<std::int64_t>::max();

        for(int i = 0; i < its; ++i)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            bool success = model->predict(dispatcher, in, out);
            auto elapsedTime = std::chrono::high_resolution_clock::now() - startTime;

            if(success)
            {
                std::int64_t elapsedMcs = std::int64_t(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count());
                minElapsedMcs = std::min(minElapsedMcs, elapsedMcs);
            }
            else
            {
                std::cerr << "pocket-tensor model predict failed: " << modelPath << std::endl;
                return;
            }
        }

        std::cout << "pocket-tensor " << modelFileName << " elapsed mcs: " << minElapsedMcs << std::endl;
    }
}

void testPocketTensorMnist(int its)
{
    pt::Tensor in{28, 28, 1};
    in.setData({0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.328125f, 0.72265625f, 0.62109375f, 0.58984375f, 0.234375f, 0.140625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.8671875f, 0.9921875f, 0.9921875f, 0.9921875f, 0.9921875f, 0.94140625f, 0.7734375f, 0.7734375f, 0.7734375f, 0.7734375f, 0.7734375f, 0.7734375f, 0.7734375f, 0.7734375f, 0.6640625f, 0.203125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.26171875f, 0.4453125f, 0.28125f, 0.4453125f, 0.63671875f, 0.88671875f, 0.9921875f, 0.87890625f, 0.9921875f, 0.9921875f, 0.9921875f, 0.9765625f, 0.89453125f, 0.9921875f, 0.9921875f, 0.546875f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.06640625f, 0.2578125f, 0.0546875f, 0.26171875f, 0.26171875f, 0.26171875f, 0.23046875f, 0.08203125f, 0.921875f, 0.9921875f, 0.4140625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.32421875f, 0.98828125f, 0.81640625f, 0.0703125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.0859375f, 0.91015625f, 0.99609375f, 0.32421875f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.50390625f, 0.9921875f, 0.9296875f, 0.171875f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.23046875f, 0.97265625f, 0.9921875f, 0.2421875f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.51953125f, 0.9921875f, 0.73046875f, 0.01953125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.03515625f, 0.80078125f, 0.96875f, 0.2265625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.4921875f, 0.9921875f, 0.7109375f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.29296875f, 0.98046875f, 0.9375f, 0.22265625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.07421875f, 0.86328125f, 0.9921875f, 0.6484375f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.01171875f, 0.79296875f, 0.9921875f, 0.85546875f, 0.13671875f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.1484375f, 0.9921875f, 0.9921875f, 0.30078125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.12109375f, 0.875f, 0.9921875f, 0.44921875f, 0.00390625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.51953125f, 0.9921875f, 0.9921875f, 0.203125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.23828125f, 0.9453125f, 0.9921875f, 0.9921875f, 0.203125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.47265625f, 0.9921875f, 0.9921875f, 0.85546875f, 0.15625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.47265625f, 0.9921875f, 0.80859375f, 0.0703125f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    testPocketTensorModel("mnist", in, its);
}

void testPocketTensorImdb(int its)
{
    pt::Tensor in{80};
    in.setData({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 591, 202, 14, 31, 6, 717, 10, 10, 18142, 10698, 5, 4, 360, 7, 4, 177, 5760, 394, 354, 4, 123, 9, 1035, 1035, 1035, 10, 10, 13, 92, 124, 89, 488, 7944, 100, 28, 1668, 14, 31, 23, 27, 7479, 29, 220, 468, 8, 124, 14, 286, 170, 8, 157, 46, 5, 27, 239, 16, 179, 15387, 38, 32, 25, 7944, 451, 202, 14, 6, 717});
    testPocketTensorModel("imdb", in, its);
}
