/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_tensor.h"

#include <numeric>
#include "pt_parser.h"
#include "pt_add.h"
#include "pt_multiply.h"
#include "pt_multiply_add.h"

namespace pt
{

namespace
{
    template<class AddType>
    void addImpl(const Tensor& a, Tensor& out) noexcept
    {
        auto aBegin = a.begin();
        auto outBegin = out.begin();
        AddType()(&*aBegin, &*outBegin, int(a.getSize()));
    }

    template<class MultiplyType>
    void multiplyImpl(const Tensor& a, Tensor& out) noexcept
    {
        auto aBegin = a.begin();
        auto outBegin = out.begin();
        MultiplyType()(&*aBegin, &*outBegin, int(a.getSize()));
    }

    template<class MultiplyAddType>
    void dotImpl(const Tensor& a, const Tensor& b, Tensor& out) noexcept
    {
        auto aIt = a.begin();
        auto bBegin = b.begin();
        MultiplyAddType multiplyAdd;

        auto iInc = int(a.getDims()[1]);
        auto outInc = int(out.getDims()[1]);
        auto outUnpaddedInc = outInc;

        if(! out.getUnpaddedDims().empty())
        {
            outUnpaddedInc = int(out.getUnpaddedDims()[1]);
        }

        for(auto outIt = out.begin(), outEnd = out.end(); outIt != outEnd; outIt += outInc)
        {
            auto bIt = bBegin;

            for(auto outIt2 = outIt; outIt2 != outIt + outUnpaddedInc; ++outIt2)
            {
                *outIt2 = multiplyAdd(&*aIt, &*bIt, iInc);
                bIt += iInc;
            }

            aIt += iInc;
        }
    }

    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& scale, const Tensor& in, Tensor& out) noexcept
    {
        auto inBegin = in.begin();
        auto outBegin = out.begin();
        auto sBegin = scale.begin();
        MultiplyAddType()(&*inBegin, &*sBegin, &*outBegin, int(in.getSize()));
    }
}

std::unique_ptr<Tensor> Tensor::create(std::size_t dims, std::istream& stream)
{
    if(dims == 0)
    {
        PT_LOG_ERROR << "Invalid dims value: " << dims << std::endl;
        return std::unique_ptr<Tensor>();
    }

    std::unique_ptr<Tensor> tensor(new Tensor());
    tensor->_dims.reserve(dims);

    for(std::size_t i = 0; i != dims; ++i)
    {
        unsigned int stride = 0;

        if(! Parser::parse(stream, stride))
        {
            PT_LOG_ERROR << "Stride parse failed" << std::endl;
            return std::unique_ptr<Tensor>();
        }

        if(stride == 0)
        {
            PT_LOG_ERROR << "Invalid stride value: " << stride << std::endl;
            return std::unique_ptr<Tensor>();
        }

        tensor->_dims.push_back(stride);
    }

    std::size_t size = tensor->getSize();

    #if PT_DOUBLE_ENABLE
        std::vector<float> data(size);
        tensor->_data.resize(size);

        if(! Parser::parse(stream, data.data(), size))
        {
            PT_LOG_ERROR << "Data parse failed" << std::endl;
            return std::unique_ptr<Tensor>();
        }

        for(std::size_t index = 0; index != size; ++index)
        {
            tensor->_data[index] = FloatType(data[index]);
        }
    #else
        tensor->_data.resize(size);

        if(! Parser::parse(stream, tensor->_data.data(), size))
        {
            PT_LOG_ERROR << "Data parse failed" << std::endl;
            return std::unique_ptr<Tensor>();
        }
    #endif

    return tensor;
}

void Tensor::copyTo(Tensor& other) const
{
    other._dims.clear();
    other._dims.reserve(_dims.size());
    other._dims.insert(other._dims.end(), _dims.begin(), _dims.end());

    other._unpaddedDims.clear();
    other._unpaddedDims.reserve(_unpaddedDims.size());
    other._unpaddedDims.insert(other._unpaddedDims.end(), _unpaddedDims.begin(), _unpaddedDims.end());

    other._data.clear();
    other._data.reserve(_data.size());
    other._data.insert(other._data.end(), _data.begin(), _data.end());
}

void Tensor::resize(std::size_t i)
{
    PT_ASSERT(i > 0);

    bool repad = hasPadding();

    if(repad)
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.push_back(i);

    _data.resize(i);

    if(repad)
    {
        addPadding(false);
    }
}

void Tensor::resize(std::size_t i, std::size_t j)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);

    bool repad = hasPadding();

    if(repad)
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.reserve(2);
    _dims.push_back(i);
    _dims.push_back(j);

    _data.resize(i * j);

    if(repad)
    {
        addPadding(false);
    }
}

void Tensor::resize(std::size_t i, std::size_t j, std::size_t k)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);

    bool repad = hasPadding();

    if(repad)
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.reserve(3);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);

    _data.resize(i * j * k);

    if(repad)
    {
        addPadding(false);
    }
}

void Tensor::resize(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);
    PT_ASSERT(l > 0);

    bool repad = hasPadding();

    if(repad)
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.reserve(4);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);
    _dims.push_back(l);

    _data.resize(i * j * k * l);

    if(repad)
    {
        addPadding(false);
    }
}

void Tensor::resizeWithPadding(std::size_t i)
{
    PT_ASSERT(i > 0);

    if(hasPadding())
    {
        removePadding(false);
    }

    _unpaddedDims.clear();
    _dims.clear();
    _dims.push_back(i);

    addPadding(false);

    _data.resize(getSize());
}

void Tensor::resizeWithPadding(std::size_t i, std::size_t j)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);

    if(hasPadding())
    {
        removePadding(false);
    }

    _unpaddedDims.clear();
    _dims.clear();
    _dims.reserve(2);
    _dims.push_back(i);
    _dims.push_back(j);

    addPadding(false);

    _data.resize(getSize());
}

void Tensor::resizeWithPadding(std::size_t i, std::size_t j, std::size_t k)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);

    if(hasPadding())
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.reserve(3);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);

    addPadding(false);

    _data.resize(getSize());
}

void Tensor::resizeWithPadding(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);
    PT_ASSERT(l > 0);

    if(hasPadding())
    {
        removePadding(false);
    }

    _dims.clear();
    _dims.reserve(4);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);
    _dims.push_back(l);

    addPadding(false);

    _data.resize(getSize());
}

void Tensor::addPadding(bool copyData)
{
    PT_ASSERT(isValid());

    if(! hasPadding())
    {
        std::size_t numDims = _dims.size();
        _unpaddedDims.reserve(_dims.size());
        _unpaddedDims.insert(_unpaddedDims.end(), _dims.begin(), _dims.end());

        std::size_t unpaddedLastDim = _dims[numDims - 1];
        std::size_t lastDimMod = unpaddedLastDim % VectorSize;

        if(lastDimMod)
        {
            std::size_t paddedLastDim = unpaddedLastDim + VectorSize - lastDimMod;
            _dims[numDims - 1] = paddedLastDim;

            std::size_t paddedSize = getSize();

            if(copyData && numDims > 1)
            {
                DataVector paddedDataVector(paddedSize);
                const Type* unpaddedData = _data.data();
                Type* paddedData = paddedDataVector.data();
                std::size_t padding = paddedLastDim - unpaddedLastDim;

                for(std::size_t block = 0, blockLimit = paddedSize / paddedLastDim; block != blockLimit; ++block)
                {
                    std::memcpy(paddedData, unpaddedData, unpaddedLastDim * sizeof(Type));
                    unpaddedData += unpaddedLastDim;
                    paddedData += unpaddedLastDim;

                    std::memset(paddedData, 0, padding * sizeof(Type));
                    paddedData += padding;
                }

                _data = std::move(paddedDataVector);
            }
            else
            {
                _data.resize(paddedSize, 0);
            }
        }
    }
}

void Tensor::removePadding(bool copyData)
{
    PT_ASSERT(isValid());

    if(hasPadding())
    {
        std::size_t paddedSize = getSize();
        std::size_t numDims = _dims.size();
        std::size_t paddedLastDim = _dims[numDims - 1];
        std::size_t unpaddedLastDim = _unpaddedDims[numDims - 1];

        _dims.clear();
        _dims.reserve(_unpaddedDims.size());
        _dims.insert(_dims.end(), _unpaddedDims.begin(), _unpaddedDims.end());

        _unpaddedDims.clear();

        if(paddedLastDim != unpaddedLastDim)
        {
            std::size_t unpaddedSize = getSize();

            if(copyData && numDims > 1)
            {
                DataVector unpaddedDataVector(unpaddedSize);
                const Type* paddedData = _data.data();
                Type* unpaddedData = unpaddedDataVector.data();

                for(std::size_t block = 0, blockLimit = paddedSize / paddedLastDim; block != blockLimit; ++block)
                {
                    std::memcpy(unpaddedData, paddedData, unpaddedLastDim * sizeof(Type));
                    unpaddedData += unpaddedLastDim;
                    paddedData += paddedLastDim;
                }

                _data = std::move(unpaddedDataVector);
            }
            else
            {
                _data.resize(unpaddedSize);
            }
        }
    }
}

void Tensor::fill(Type value) noexcept
{
    std::fill(begin(), end(), value);
}

void Tensor::flatten()
{
    PT_ASSERT(isValid());

    bool repad = hasPadding();
    removePadding();

    auto size = getSize();
    _dims.clear();
    _dims.push_back(size);

    if(repad)
    {
        addPadding();
    }
}

void Tensor::unpack(std::size_t row, Tensor& out) const
{
    PT_ASSERT(isValid());
    PT_ASSERT(_dims.size() >= 2);
    PT_ASSERT(row < _dims[0]);

    auto packSize = std::accumulate(_dims.begin() + 1, _dims.end(), std::size_t(0));
    auto base = row * packSize;
    auto first = begin() + long(base);
    auto last = first + packSize;

    out._dims.clear();
    out._dims.reserve(_dims.size() - 1);
    out._dims.insert(out._dims.end(), _dims.begin() + 1, _dims.end());

    out._unpaddedDims.clear();

    if(! _unpaddedDims.empty())
    {
        out._unpaddedDims.reserve(_unpaddedDims.size() - 1);
        out._unpaddedDims.insert(out._unpaddedDims.end(), _unpaddedDims.begin() + 1, _unpaddedDims.end());
    }

    out._data.clear();
    out._data.reserve(std::size_t(last - first));
    out._data.insert(out._data.end(), first, last);
}

void Tensor::select(std::size_t row, Tensor& out) const
{
    unpack(row, out);
    out._dims.insert(out._dims.begin(), 1);

    if(! out._unpaddedDims.empty())
    {
        out._unpaddedDims.insert(out._unpaddedDims.begin(), 1);
    }
}

void Tensor::add(const Tensor& other, Tensor& out) const
{
    PT_ASSERT(_dims == other._dims);

    auto size = getSize();
    copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && size % (Tensor::VectorSize * 2) == 0)
    {
        addImpl<Vector2Add>(other, out);
    }
    else if(size % Tensor::VectorSize == 0)
    {
        addImpl<VectorAdd>(other, out);
    }
    else
    {
        addImpl<ScalarAdd>(other, out);
    }
}

void Tensor::multiply(const Tensor& other, Tensor& out) const
{
    PT_ASSERT(isValid());
    PT_ASSERT(_dims == other._dims);

    auto size = getSize();
    copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && size % (Tensor::VectorSize * 2) == 0)
    {
        multiplyImpl<Vector2Multiply>(other, out);
    }
    else if(size % Tensor::VectorSize == 0)
    {
        multiplyImpl<VectorMultiply>(other, out);
    }
    else
    {
        multiplyImpl<ScalarMultiply>(other, out);
    }
}

void Tensor::dot(const Tensor& other, Tensor& out) const
{
    PT_ASSERT(_dims.size() == 2);
    PT_ASSERT(other._dims.size() == 2);
    PT_ASSERT(_dims[1] == other._dims[1]);

    if(_unpaddedDims.empty())
    {
        PT_ASSERT(other._unpaddedDims.empty());

        if(out.hasPadding())
        {
            out.removePadding(false);
        }

        out.resize(_dims[0], other._dims[0]);
    }
    else
    {
        PT_ASSERT(_unpaddedDims.size() == 2);
        PT_ASSERT(other._unpaddedDims.size() == 2);
        PT_ASSERT(_unpaddedDims[1] == other._unpaddedDims[1]);

        out.resizeWithPadding(_unpaddedDims[0], other._unpaddedDims[0]);
    }

    auto size = int(_dims[1]);

    if(PT_LOOP_UNROLLING_ENABLE && size % (Tensor::VectorSize * 2) == 0)
    {
        dotImpl<Vector2MultiplyAdd>(*this, other, out);
    }
    else if(size % Tensor::VectorSize == 0)
    {
        dotImpl<VectorMultiplyAdd>(*this, other, out);
    }
    else
    {
        dotImpl<ScalarMultiplyAdd>(*this, other, out);
    }
}

void Tensor::fma(const Tensor& scale, const Tensor& bias, Tensor& out) const
{
    PT_ASSERT(_dims == scale._dims);
    PT_ASSERT(_dims == bias._dims);

    auto size = getSize();
    bias.copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && size % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(scale, *this, out);
    }
    else if(size % Tensor::VectorSize == 0)
    {
        multiplyAddImpl<VectorMultiplyAdd>(scale, *this, out);
    }
    else
    {
        multiplyAddImpl<ScalarMultiplyAdd>(scale, *this, out);
    }
}

void Tensor::eraseDummyDims() noexcept
{
    auto numDims = _dims.size();

    if(numDims > 1)
    {
        for(std::size_t index = 0; index != numDims - 1; ++index)
        {
            if(_dims[index] == 1)
            {
                _dims.erase(_dims.begin() + long(index));

                if(! _unpaddedDims.empty())
                {
                    _unpaddedDims.erase(_unpaddedDims.begin() + long(index));
                }

                --index;
                --numDims;
            }
        }
    }
}

void Tensor::clear() noexcept
{
    _dims.clear();
    _unpaddedDims.clear();
    _data.clear();
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor)
{
    const auto& dims = tensor.getDims();
    std::vector<std::size_t> steps(dims.size());
    std::partial_sum(dims.rbegin(), dims.rend(), steps.rbegin(), std::multiplies<std::size_t>());

    size_t count = 0;

    for(auto value : tensor.getData())
    {
        for(std::size_t step : steps)
        {
            if(count % step == 0)
            {
                stream << '[';
            }
        }

        stream << value;
        ++count;

        for(std::size_t step : steps)
        {
            if(count % step == 0)
            {
                stream << ']';
            }
        }

        if(count != steps[0])
        {
            stream << ", ";
        }
    }

    return stream;
}

}
