/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_ELU_LAYER_H
#define PT_ELU_LAYER_H

#include "pt_libsimdpp.h"
#include "pt_layer.h"

namespace pt
{

class EluLayer : public Layer
{

public:
    static std::unique_ptr<EluLayer> create(std::istream& stream);

    bool apply(const Config& config, Tensor&& in, Tensor& out) const final;

protected:
    FloatType _alpha;

    explicit EluLayer(FloatType alpha) noexcept;
};

}

#endif
