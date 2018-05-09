#include "embnet_flatten_layer.h"

#include "embnet_tensor.h"

namespace embnet
{

class FixedFlattenLayer : public FixedLayer
{

public:
    FixedFlattenLayer() = default;

    bool apply(const Config&, FixedTensor&& in, FixedTensor& out) const final
    {
        out = std::move(in);
        out.flatten();
        return true;
    }
};

bool FlattenLayer::apply(const Config&, Tensor&& in, Tensor& out) const
{
    out = std::move(in);
    out.flatten();
    return true;
}

std::unique_ptr<FixedLayer> FlattenLayer::getFixedLayer()
{
    return std::unique_ptr<FixedLayer>(new FixedFlattenLayer());
}

}
