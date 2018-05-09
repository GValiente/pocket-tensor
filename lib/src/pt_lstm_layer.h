/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LSTM_LAYER_H
#define PT_LSTM_LAYER_H

#include "pt_tensor.h"
#include "pt_activation_layer.h"

namespace pt
{

class LstmLayer : public Layer
{

public:
    static std::unique_ptr<LstmLayer> create(std::istream& stream);

    bool apply(const Config& config, Tensor&& in, Tensor& out) const final;

protected:
    struct TempData;

    Tensor _wi;
    Tensor _ui;
    Tensor _bi;
    Tensor _wf;
    Tensor _uf;
    Tensor _bf;
    Tensor _wc;
    Tensor _uc;
    Tensor _bc;
    Tensor _wo;
    Tensor _uo;
    Tensor _bo;
    std::unique_ptr<ActivationLayer> _innerActivation;
    std::unique_ptr<ActivationLayer> _activation;
    bool _returnSequences;

    LstmLayer(Tensor&& wi, Tensor&& ui, Tensor&& bi, Tensor&& wf, Tensor&& uf, Tensor&& bf,
          Tensor&& wc, Tensor&& uc, Tensor&& bc, Tensor&& wo, Tensor&& uo, Tensor&& bo,
	      std::unique_ptr<ActivationLayer>&& innerActivation,
	      std::unique_ptr<ActivationLayer>&& activation, bool returnSequences) noexcept;

    void _step(const Config& config, TempData& tempData, Tensor& out) const;
};

}

#endif
