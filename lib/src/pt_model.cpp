/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_model.h"

#include <string>
#include <fstream>
#include "pt_parser.h"
#include "pt_tensor.h"

namespace pt
{

namespace
{
    template<class LayerType, class TensorType>
    bool predictImpl(const Config& config, const std::vector<std::unique_ptr<LayerType>>& layers,
                     TensorType in, TensorType& out)
    {
        if(! in.isValid())
        {
            PT_LOG_ERROR << "Input tensor is not valid" << std::endl;
            return false;
        }

        std::size_t layersCount = layers.size();
        in.addPadding();

        for(std::size_t i = 0; i != layersCount - 1; ++i)
        {
            if(! layers[i]->apply(config, std::move(in), out))
            {
                PT_LOG_ERROR << "Layer apply failed" << std::endl;
                return false;
            }

            in = std::move(out);
        }

        if(! layers[layersCount - 1]->apply(config, std::move(in), out))
        {
            PT_LOG_ERROR << "Layer apply failed" << std::endl;
            return false;
        }

        out.removePadding();
        return true;
    }
}

std::unique_ptr<Model> Model::create(const std::string& filePath)
{
    std::ifstream stream(filePath, std::ios::binary);

    if(! stream.good())
    {
        PT_LOG_ERROR << "File open failed: " << filePath << std::endl;
        return std::unique_ptr<Model>();
    }

    auto model = create(stream);

    if(! model)
    {
        PT_LOG_ERROR << "File parse failed: " << filePath << std::endl;
        return std::unique_ptr<Model>();
    }

    return model;
}

std::unique_ptr<Model> Model::create(std::istream& stream)
{
    unsigned int layersCount = 0;

    if(! Parser::parse(stream, layersCount))
    {
        PT_LOG_ERROR << "Layers count parse failed" << std::endl;
        return std::unique_ptr<Model>();
    }

    if(! layersCount)
    {
        PT_LOG_ERROR << "Invalid layers count: " << layersCount << std::endl;
        return std::unique_ptr<Model>();
    }

    std::vector<std::unique_ptr<Layer>> layers;

    for(unsigned int i = 0; i != layersCount; ++i)
    {
        auto layer = Layer::create(stream);

        if(! layer)
        {
            PT_LOG_ERROR << "Layer parse failed" << std::endl;
            return std::unique_ptr<Model>();
        }

        layers.push_back(std::move(layer));
    }

    return std::unique_ptr<Model>(new Model(std::move(layers)));
}

bool Model::predict(Tensor in, Tensor& out) const
{
    return predictImpl(_config, _layers, std::move(in), out);
}

Model::Model(std::vector<std::unique_ptr<Layer>>&& layers) noexcept :
    _layers(std::move(layers))
{
}

}
