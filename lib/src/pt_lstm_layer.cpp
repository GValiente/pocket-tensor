/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_lstm_layer.h"

#include "pt_parser.h"
#include "pt_logger.h"

namespace pt
{

struct LstmLayer::TempData
{
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

    void dot(const Tensor& w, const Tensor& b, const Tensor& u, Tensor& out)
    {
        inRow.dot(w, tmp1);
        tmp1.add(b, tmp2);
        ht.dot(u, tmp1);
        tmp2.add(tmp1, out);
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

bool LstmLayer::apply(const Config& config, Tensor&& in, Tensor& out) const
{
    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{in.getUnpaddedDims()} << ")" << std::endl;
        return false;
    }

    auto outDim = _bo.getUnpaddedDims()[1];
    auto steps = iw[0];

    TempData tempData;
    tempData.ht.resizeWithPadding(1, outDim);
    tempData.ct.resizeWithPadding(1, outDim);

    if(_returnSequences)
    {
        out.resizeWithPadding(steps, outDim);

        auto outIt = out.begin();

        for(std::size_t s = 0; s != steps; ++s)
        {
            in.select(s, tempData.inRow);
            _step(config, tempData, tempData.last);
            std::copy(tempData.last.begin(), tempData.last.end(), outIt);
            outIt += tempData.last.end() - tempData.last.begin();
        }
    }
    else
    {
        for(std::size_t s = 0; s != steps; ++s)
        {
            in.select(s, tempData.inRow);
            _step(config, tempData, out);
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
    _wi.addPadding();
    _ui.addPadding();
    _bi.addPadding();
    _wf.addPadding();
    _uf.addPadding();
    _bf.addPadding();
    _wc.addPadding();
    _uc.addPadding();
    _bc.addPadding();
    _wo.addPadding();
    _uo.addPadding();
    _bo.addPadding();
}

void LstmLayer::_step(const Config& config, TempData& tempData, Tensor& out) const
{
    tempData.dot(_wi, _bi, _ui, tempData.i);
    tempData.dot(_wf, _bf, _uf, tempData.f);
    tempData.dot(_wc, _bc, _uc, tempData.c);
    tempData.dot(_wo, _bo, _uo, tempData.o);

    _innerActivation->apply(config, tempData.i);
    _innerActivation->apply(config, tempData.f);
    _activation->apply(config, tempData.c);
    _innerActivation->apply(config, tempData.o);

    tempData.f.multiply(tempData.ct, tempData.tmp1);
    tempData.i.multiply(tempData.c, tempData.tmp2);
    tempData.tmp1.add(tempData.tmp2, tempData.ct);

    tempData.ct.copyTo(tempData.c);
    _activation->apply(config, tempData.c);

    tempData.o.multiply(tempData.c, tempData.ht);
    tempData.ht.copyTo(out);
}

}
