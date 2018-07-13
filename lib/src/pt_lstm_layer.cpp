/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_lstm_layer.h"

#include "pt_parser.h"
#include "pt_dispatcher.h"
#include "pt_layer_data.h"
#include "pt_logger.h"

namespace pt
{

struct LstmLayer::TempData
{
    Dispatcher dummyDispatcher;
    Tensor ht;
    Tensor ct;
    Tensor last;
    Tensor inRow;
    Tensor i;
    Tensor f;
    Tensor c;
    Tensor o;
    Tensor tmp1;
    Tensor tmp2;

    explicit TempData(std::size_t outDim) :
        dummyDispatcher(1)
    {
        ht.resize(1, outDim);
        ct.resize(1, outDim);
    }

    void dot(const Tensor& w, const Tensor& b, const Tensor& u, Tensor& out)
    {
        inRow.dot(w, tmp1, dummyDispatcher);
        tmp1.add(b, tmp2, dummyDispatcher);
        ht.dot(u, tmp1, dummyDispatcher);
        tmp2.add(tmp1, out, dummyDispatcher);
    }
};

std::unique_ptr<LstmLayer> LstmLayer::create(std::istream& stream)
{
    auto wi = Tensor::create(2, stream);

    if(! wi)
    {
        PT_LOG_ERROR << "wi tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto ui = Tensor::create(2, stream);

    if(! ui)
    {
        PT_LOG_ERROR << "ui tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto bi = Tensor::create(2, stream);

    if(! bi)
    {
        PT_LOG_ERROR << "bi tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto wf = Tensor::create(2, stream);

    if(! wf)
    {
        PT_LOG_ERROR << "wf tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto uf = Tensor::create(2, stream);

    if(! uf)
    {
        PT_LOG_ERROR << "uf tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto bf = Tensor::create(2, stream);

    if(! bf)
    {
        PT_LOG_ERROR << "bf tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto wc = Tensor::create(2, stream);

    if(! wc)
    {
        PT_LOG_ERROR << "wc tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto uc = Tensor::create(2, stream);

    if(! uc)
    {
        PT_LOG_ERROR << "uc tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto bc = Tensor::create(2, stream);

    if(! bc)
    {
        PT_LOG_ERROR << "bc tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto wo = Tensor::create(2, stream);

    if(! wo)
    {
        PT_LOG_ERROR << "wo tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto uo = Tensor::create(2, stream);

    if(! uo)
    {
        PT_LOG_ERROR << "uo tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto bo = Tensor::create(2, stream);

    if(! bo)
    {
        PT_LOG_ERROR << "bo tensor parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto innerActivation = ActivationLayer::create(stream);

    if(! innerActivation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    auto activation = ActivationLayer::create(stream);

    if(! activation)
    {
        PT_LOG_ERROR << "Activation layer parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    unsigned int returnSequences = 0;

    if(! Parser::parse(stream, returnSequences))
    {
        PT_LOG_ERROR << "Return sequences parse failed" << std::endl;
        return std::unique_ptr<LstmLayer>();
    }

    return std::unique_ptr<LstmLayer>(new LstmLayer(std::move(*wi), std::move(*ui), std::move(*bi),
                                                    std::move(*wf), std::move(*uf), std::move(*bf),
                                                    std::move(*wc), std::move(*uc), std::move(*bc),
                                                    std::move(*wo), std::move(*uo), std::move(*bo),
                                                    std::move(innerActivation),
                                                    std::move(activation), returnSequences));
}

bool LstmLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    auto outDim = _bo.getDims()[1];
    auto steps = iw[0];

    TempData tempData(outDim);
    Tensor& out = layerData.out;

    if(_returnSequences)
    {
        out.resize(steps, outDim);

        auto outIt = out.begin();

        for(std::size_t s = 0; s != steps; ++s)
        {
            in.select(s, tempData.inRow);
            _step(tempData, tempData.last);
            std::copy(tempData.last.begin(), tempData.last.end(), outIt);
            outIt += tempData.last.end() - tempData.last.begin();
        }
    }
    else
    {
        for(std::size_t s = 0; s != steps; ++s)
        {
            in.select(s, tempData.inRow);
            _step(tempData, out);
        }
    }

    out.eraseDummyDims();
    return true;
}

LstmLayer::LstmLayer(Tensor&& wi, Tensor&& ui, Tensor&& bi, Tensor&& wf, Tensor&& uf, Tensor&& bf,
                     Tensor&& wc, Tensor&& uc, Tensor&& bc, Tensor&& wo, Tensor&& uo, Tensor&& bo,
                     std::unique_ptr<ActivationLayer>&& innerActivation,
                     std::unique_ptr<ActivationLayer>&& activation, bool returnSequences) noexcept :
    _wi(std::move(wi)),
    _ui(std::move(ui)),
    _bi(std::move(bi)),
    _wf(std::move(wf)),
    _uf(std::move(uf)),
    _bf(std::move(bf)),
    _wc(std::move(wc)),
    _uc(std::move(uc)),
    _bc(std::move(bc)),
    _wo(std::move(wo)),
    _uo(std::move(uo)),
    _bo(std::move(bo)),
    _innerActivation(std::move(innerActivation)),
    _activation(std::move(activation)),
    _returnSequences(returnSequences)
{
}

void LstmLayer::_step(TempData& tempData, Tensor& out) const
{
    tempData.dot(_wi, _bi, _ui, tempData.i);
    _innerActivation->apply(tempData.i);

    tempData.dot(_wf, _bf, _uf, tempData.f);
    _innerActivation->apply(tempData.f);

    tempData.dot(_wc, _bc, _uc, tempData.c);
    _activation->apply(tempData.c);

    tempData.dot(_wo, _bo, _uo, tempData.o);
    _innerActivation->apply(tempData.o);

    // join

    tempData.f.multiply(tempData.ct, tempData.tmp1, tempData.dummyDispatcher);

    tempData.i.multiply(tempData.c, tempData.tmp2, tempData.dummyDispatcher);

    // join
    // seq

    tempData.tmp1.add(tempData.tmp2, tempData.ct, tempData.dummyDispatcher);

    tempData.ct.copyTo(tempData.c);
    _activation->apply(tempData.c);

    tempData.o.multiply(tempData.c, tempData.ht, tempData.dummyDispatcher);
    tempData.ht.copyTo(out);
}

}
