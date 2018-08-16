/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_activation_layer.h"

#include "pt_parser.h"
#include "pt_layer_data.h"
#include "pt_linear_activation_layer.h"
#include "pt_relu_activation_layer.h"
#include "pt_elu_activation_layer.h"
#include "pt_soft_plus_activation_layer.h"
#include "pt_soft_sign_activation_layer.h"
#include "pt_sigmoid_activation_layer.h"
#include "pt_tanh_activation_layer.h"
#include "pt_hard_sigmoid_activation_layer.h"
#include "pt_soft_max_activation_layer.h"
#include "pt_selu_activation_layer.h"

namespace pt
{

namespace
{
    enum ActivationType
    {
        Linear = 1,
        Relu = 2,
        Elu = 3,
        SoftPlus = 4,
        SoftSign = 5,
        Sigmoid = 6,
        Tanh = 7,
        HardSigmoid = 8,
        SoftMax = 9,
        Selu = 10
    };
}

std::unique_ptr<ActivationLayer> ActivationLayer::create(std::istream& stream)
{
    unsigned int activationLayerID = 0;

    if(! Parser::parse(stream, activationLayerID))
    {
        PT_LOG_ERROR << "Activation ID parse failed" << std::endl;
        return std::unique_ptr<ActivationLayer>();
    }

    std::unique_ptr<ActivationLayer> activationLayer;

    switch(activationLayerID)
    {

    case Linear:
        activationLayer.reset(new LinearActivationLayer());
        break;

    case Relu:
        activationLayer.reset(new ReluActivationLayer());
        break;

    case Elu:
        activationLayer.reset(new EluActivationLayer());
        break;

    case SoftPlus:
        activationLayer.reset(new SoftPlusActivationLayer());
        break;

    case SoftSign:
        activationLayer.reset(new SoftSignActivationLayer());
        break;

    case Sigmoid:
        activationLayer.reset(new SigmoidActivationLayer());
        break;

    case Tanh:
        activationLayer.reset(new TanhActivationLayer());
        break;

    case HardSigmoid:
        activationLayer.reset(new HardSigmoidActivationLayer());
        break;

    case SoftMax:
        activationLayer.reset(new SoftMaxActivationLayer());
        break;

    case Selu:
        activationLayer.reset(new SeluActivationLayer());
        break;

    default:
        PT_LOG_ERROR << "Unknown activation layer ID: " << activationLayerID << std::endl;
    }

    return activationLayer;
}

bool ActivationLayer::apply(LayerData& layerData) const
{
    layerData.out = std::move(layerData.in);
    apply(layerData.out);
    return true;
}

}
