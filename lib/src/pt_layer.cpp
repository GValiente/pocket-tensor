/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_layer.h"

#include "pt_parser.h"
#include "pt_dense_layer.h"
#include "pt_conv_1d_layer.h"
#include "pt_conv_2d_layer.h"
#include "pt_locally_connected_1d_layer.h"
#include "pt_flatten_layer.h"
#include "pt_elu_layer.h"
#include "pt_activation_layer.h"
#include "pt_max_pooling_2d_layer.h"
#include "pt_lstm_layer.h"
#include "pt_embedding_layer.h"
#include "pt_batch_normalization_layer.h"
#include "pt_leaky_relu_layer.h"
#include "pt_global_max_pooling_2d_layer.h"

namespace pt
{

namespace
{
    enum LayerType
    {
        Dense = 1,
        Conv1D = 2,
        Conv2D = 3,
        LocallyConnected1D = 4,
        Flatten = 6,
        Elu = 7,
        Activation = 8,
        MaxPooling2D = 9,
        Lstm = 10,
        Embedding = 11,
        BatchNormalization = 12,
        LeakyRelu = 13,
        GlobalMaxPooling2D = 14
    };
}

std::unique_ptr<Layer> Layer::create(std::istream& stream)
{
    unsigned int layerID = 0;

    if(! Parser::parse(stream, layerID))
    {
        PT_LOG_ERROR << "Layer ID parse failed" << std::endl;
        return nullptr;
    }

    std::unique_ptr<Layer> layer;

    switch(layerID)
    {

    case Dense:
        layer = DenseLayer::create(stream);
        break;

    case Conv1D:
        layer = Conv1DLayer::create(stream);
        break;

    case Conv2D:
        layer = Conv2DLayer::create(stream);
        break;

    case LocallyConnected1D:
        layer = LocallyConnected1DLayer::create(stream);
        break;

    case Flatten:
        layer.reset(new FlattenLayer());
        break;

    case Elu:
        layer = EluLayer::create(stream);
        break;

    case Activation:
        layer = ActivationLayer::create(stream);
        break;

    case MaxPooling2D:
        layer = MaxPooling2DLayer::create(stream);
        break;

    case Lstm:
        layer = LstmLayer::create(stream);
        break;

    case Embedding:
        layer = EmbeddingLayer::create(stream);
        break;

    case BatchNormalization:
        layer = BatchNormalizationLayer::create(stream);
        break;

    case LeakyRelu:
        layer = LeakyReluLayer::create(stream);
        break;

    case GlobalMaxPooling2D:
        layer = GlobalMaxPooling2DLayer::create(stream);
        break;

    default:
        PT_LOG_ERROR << "Unknown layer ID: " << layerID << std::endl;
    }

    return layer;
}

Layer::~Layer() noexcept
{
}

}
