/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_SOFT_MAX_ACTIVATION_LAYER_H
#define PT_SOFT_MAX_ACTIVATION_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class SoftMaxActivationLayer : public ActivationLayer
{

public:
    using ActivationLayer::apply;

    SoftMaxActivationLayer() = default;

    void apply(const Config&, Tensor& out) const final
    {
        FloatType d = 0;

        for(FloatType& value : out)
        {
            value = std::exp(value);
            d += value;
        }

        Tensor::Vector vd = makeVector(1 / d);

        for(auto it = out.begin(), end = out.end(); it != end; it += Tensor::VectorSize)
        {
            auto ptr = &*it;
            Tensor::Vector v = simdpp::load(ptr);
            simdpp::store(ptr, simdpp::mul(v, vd));
        }
    }
};

}

#endif
