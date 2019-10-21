/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LAYER_DATA_H
#define PT_LAYER_DATA_H

#include "pt_tensor.h"

namespace pt
{

class Config;
class Dispatcher;

struct LayerData
{
    Tensor in;
    Tensor& out;
    Dispatcher& dispatcher;
    const Config& config;
};

}

#endif
