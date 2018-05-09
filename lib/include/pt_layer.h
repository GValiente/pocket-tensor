/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LAYER_H
#define PT_LAYER_H

#include <memory>
#include <iosfwd>

namespace pt
{

class Tensor;
class Config;

class Layer
{

public:
    static std::unique_ptr<Layer> create(std::istream& stream);

    virtual ~Layer() noexcept;

    virtual bool apply(const Config& config, Tensor&& in, Tensor& out) const = 0;

protected:
    Layer() = default;
};

}

#endif
