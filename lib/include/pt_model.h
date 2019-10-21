/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_MODEL_H
#define PT_MODEL_H

#include <vector>
#include <string>
#include "pt_layer.h"
#include "pt_config.h"

namespace pt
{

class Tensor;
class Dispatcher;

class Model
{

public:
    static std::unique_ptr<Model> create(const std::string& filePath);

    static std::unique_ptr<Model> create(std::istream& stream);

    bool predict(Tensor in, Tensor& out) const;

    bool predict(Dispatcher& dispatcher, Tensor in, Tensor& out) const;

    const Config& getConfig() const noexcept
    {
        return _config;
    }

    Config& getConfig() noexcept
    {
        return _config;
    }

    const std::vector<std::unique_ptr<Layer>>& getLayers() const noexcept
    {
        return _layers;
    }

protected:
    std::vector<std::unique_ptr<Layer>> _layers;
    Config _config;

    Model(std::vector<std::unique_ptr<Layer>>&& layers) noexcept;
};

}

#endif
