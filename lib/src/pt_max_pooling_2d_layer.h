/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_MAX_POOLING_2D_LAYER_H
#define PT_MAX_POOLING_2D_LAYER_H

#include "pt_layer.h"

namespace pt
{

class MaxPooling2DLayer : public Layer
{

public:
    static std::unique_ptr<MaxPooling2DLayer> create(std::istream& stream);

    bool apply(const Config& config, Tensor&& in, Tensor& out) const final;

protected:
    int _poolSizeY;
    int _poolSizeX;

    MaxPooling2DLayer(int poolSizeY, int poolSizeX) noexcept :
        _poolSizeY(poolSizeY),
        _poolSizeX(poolSizeX)
    {
    }
};

}

#endif
