/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/model.h"
#include "keras/layers/conv1d.h"
#include "keras/layers/conv2d.h"
#include "keras/layers/dense.h"
#include "keras/layers/elu.h"
#include "keras/layers/embedding.h"
#include "keras/layers/flatten.h"
#include "keras/layers/locally1d.h"
#include "keras/layers/locally2d.h"
#include "keras/layers/lstm.h"
#include "keras/layers/maxpooling2d.h"
#include "keras/layers/normalization.h"
#include <limits>
#include <utility>

namespace keras {

bool Model::load_model(const std::string& filename) noexcept
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    check(file.is_open());

    unsigned num_layers = 0;
    check(read_uint(file, num_layers));

    for (size_t i = 0; i < num_layers; ++i) {
        unsigned layer_type_index = 0;
        check(read_uint(file, layer_type_index));

        auto layer = [layer_type_index]() -> std::unique_ptr<Layer> {
            switch (layer_type_index) {
            case Dense:
                return std::make_unique<layers::Dense>();
            case Conv1D:
                return std::make_unique<layers::Conv1D>();
            case Conv2D:
                return std::make_unique<layers::Conv2D>();
            case LocallyConnected1D:
                return std::make_unique<layers::LocallyConnected1D>();
            case LocallyConnected2D:
                return std::make_unique<layers::LocallyConnected2D>();
            case Flatten:
                return std::make_unique<layers::Flatten>();
            case ELU:
                return std::make_unique<layers::ELU>();
            case Activation:
                return std::make_unique<layers::Activation>();
            case MaxPooling2D:
                return std::make_unique<layers::MaxPooling2D>();
            case LSTM:
                return std::make_unique<layers::LSTM>();
            case Embedding:
                return std::make_unique<layers::Embedding>();
            case BatchNormalization:
                return std::make_unique<layers::BatchNormalization>();
            default:
                return nullptr;
            }
        }();
        check(layer);
        check(layer->load_layer(file));

        layers_.push_back(std::move(layer));
    }
    return true;
}

bool Model::apply(const Tensor& in, Tensor& out) const noexcept
{
    Tensor temp_in, temp_out;

    temp_in = in;
    for (auto&& it : layers_) {
        check(it->apply(temp_in, temp_out));
        temp_in = temp_out;
    }
    out = temp_out;

    return true;
}

} // namespace keras
