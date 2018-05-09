#include "test_util.h"

#include <chrono>
#include <iostream>
#include "pt_model.h"

void testModel(pt::Tensor& in, const pt::Tensor& expected, const char* modelFileName, float eps)
{
    std::cout << std::fixed;

    REQUIRE(in.isValid());
    REQUIRE(modelFileName);

    auto model = pt::Model::create(std::string(PT_TEST_MODELS_FOLDER) + '/' + modelFileName + ".model");
    REQUIRE(model);

    pt::Tensor out;
    auto startTime = std::chrono::high_resolution_clock::now();
    bool success = model->predict(in, out);
    auto elapsedTime = std::chrono::high_resolution_clock::now() - startTime;
    REQUIRE(success);
    REQUIRE(out.isValid());

    for(std::size_t i = 0, l = out.getDims()[0]; i != l; ++i)
    {
        if(std::fabs(out(i) - expected(i)) >= pt::FloatType(eps))
        {
            std::cout << "Float diff: " << std::fabs(out(i) - expected(i)) << std::endl;
            REQUIRE(std::fabs(out(i) - expected(i)) < pt::FloatType(eps));
        }
    }

    auto elapsedMcs = std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count();
    std::cout << modelFileName << " elapsed mcs: " << elapsedMcs << std::endl;
}
