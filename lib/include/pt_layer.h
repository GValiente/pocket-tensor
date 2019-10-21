/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
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

struct LayerData;

class Layer
{

public:
    static std::unique_ptr<Layer> create(std::istream& stream);

    virtual ~Layer() noexcept;

    virtual bool apply(LayerData& layerData) const = 0;

protected:
    Layer() = default;
};

}

#endif
