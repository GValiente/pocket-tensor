#include "kf_bench.h"

#include <chrono>
#include <string>
#include <iostream>
#include "keras/model.h"

namespace
{
    void testKerasifyModel(const char* modelFileName, const keras::Tensor& in, int its)
    {
        auto modelPath = std::string(PT_BENCHMARK_MODELS_FOLDER) + "/kerasify/" + modelFileName + ".model";
        keras::Model model;

        if(! model.load_model(modelPath))
        {
            std::cerr << "Kerasify model load failed: " << modelPath << std::endl;
            return;
        }

        keras::Tensor out;
        std::int64_t minElapsedMcs = std::numeric_limits<std::int64_t>::max();

        for(int i = 0; i < its; ++i)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            bool success = model.apply(in, out);
            auto elapsedTime = std::chrono::high_resolution_clock::now() - startTime;

            if(success)
            {
                std::int64_t elapsedMcs = std::int64_t(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count());
                minElapsedMcs = std::min(minElapsedMcs, elapsedMcs);
            }
            else
            {
                std::cerr << "Kerasify model predict failed: " << modelPath << std::endl;
                return;
            }
        }

        std::cout << "Kerasify " << modelFileName << " elapsed mcs: " << minElapsedMcs << std::endl;
    }
}

void testKerasifyMnist(int its)
{
    keras::Tensor in(28, 28, 1);
    testKerasifyModel("mnist", in, its);
}

void testKerasifyImdb(int its)
{
    keras::Tensor in(80);
    testKerasifyModel("imdb", in, its);
}
