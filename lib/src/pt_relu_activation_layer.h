#ifndef PT_RELU_ACTIVATION_LAYER_H
#define PT_RELU_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class ReluActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    ReluActivationLayer() = default;

    void apply(Tensor& out) const final
    {
        if(out.getSize() % Tensor::VectorSize == 0)
        {
            Tensor::Vector zero = makeVector(Tensor::Type(0));

            for(auto it = out.begin(), end = out.end(); it != end; it += Tensor::VectorSize)
            {
                auto ptr = &*it;
                Tensor::Vector v = simdpp::load(ptr);
                simdpp::store(ptr, simdpp::max(v, zero));
            }
        }
        else
        {
            for(auto it = out.begin(), end = out.end(); it != end; ++it)
            {
                auto& x = *it;
                x = std::max(x, Tensor::Type(0));
            }
        }
    }
};

}

#endif
