/*
 * PocketTensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_tensor.h"

#include <array>
#include <numeric>
#include "pt_add.h"
#include "pt_multiply.h"
#include "pt_multiply_add.h"
#include "pt_parser.h"
#include "pt_dispatcher.h"

namespace pt
{

namespace
{
    template<class AddType>
    void addImpl(const Tensor& in, Tensor& out, Dispatcher& dispatcher)
    {
        struct Task
        {
            const Tensor* in;
            Tensor* out;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                int its = int(in->getSize());
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto inBegin = in->begin() + taskBegin;
                auto outBegin = out->begin() + taskBegin;
                AddType()(&*inBegin, &*outBegin, taskEnd - taskBegin);
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        int threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &in, &out, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }

    template<class MultiplyType>
    void multiplyImpl(const Tensor& in, Tensor& out, Dispatcher& dispatcher)
    {
        struct Task
        {
            const Tensor* in;
            Tensor* out;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                int its = int(in->getSize());
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto inBegin = in->begin() + taskBegin;
                auto outBegin = out->begin() + taskBegin;
                MultiplyType()(&*inBegin, &*outBegin, taskEnd - taskBegin);
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        int threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &in, &out, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }

    template<class MultiplyAddType>
    void dotImpl(const Tensor& a, const Tensor& b, Tensor& out, Dispatcher& dispatcher)
    {
        struct Task
        {
            const Tensor* a;
            const Tensor* b;
            Tensor* out;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                auto outInc = int(out->getDims()[1]);
                int its = int(out->end() - out->begin()) / outInc;
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto aIt = a->begin();
                auto iInc = int(a->getDims()[1]);
                auto bBegin = b->begin();
                auto oBegin = out->begin();
                MultiplyAddType multiplyAdd;
                aIt += taskIts * taskId * iInc;

                for(auto outIt = oBegin + (taskBegin * outInc), outEnd = oBegin + (taskEnd * outInc);
                    outIt != outEnd; outIt += outInc)
                {
                    auto bIt = bBegin;

                    for(auto outIt2 = outIt; outIt2 != outIt + outInc; ++outIt2)
                    {
                        *outIt2 = multiplyAdd(&*aIt, &*bIt, iInc);
                        bIt += iInc;
                    }

                    aIt += iInc;
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        int threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &a, &b, &out, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }

    template<class MultiplyAddType>
    void multiplyAddImpl(const Tensor& scale, const Tensor& in, Tensor& out, Dispatcher& dispatcher)
    {
        struct Task
        {
            const Tensor* scale;
            const Tensor* in;
            Tensor* out;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                int its = int(in->getSize());
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto inBegin = in->begin() + taskBegin;
                auto scaleBegin = scale->begin() + taskBegin;
                auto outBegin = out->begin() + taskBegin;
                MultiplyAddType()(&*inBegin, &*scaleBegin, &*outBegin, taskEnd - taskBegin);
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        int threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &scale, &in, &out, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
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

    other._data.clear();
    other._data.reserve(_data.size());
    other._data.insert(other._data.end(), _data.begin(), _data.end());
}

void Tensor::resize(std::size_t i)
{
    PT_ASSERT(i > 0);

    _dims.clear();
    _dims.push_back(i);
    _data.resize(i);
}

void Tensor::resize(std::size_t i, std::size_t j)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);

    _dims.clear();
    _dims.reserve(2);
    _dims.push_back(i);
    _dims.push_back(j);
    _data.resize(i * j);
}

void Tensor::resize(std::size_t i, std::size_t j, std::size_t k)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);

    _dims.clear();
    _dims.reserve(3);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);
    _data.resize(i * j * k);
}

void Tensor::resize(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    PT_ASSERT(i > 0);
    PT_ASSERT(j > 0);
    PT_ASSERT(k > 0);
    PT_ASSERT(l > 0);

    _dims.clear();
    _dims.reserve(4);
    _dims.push_back(i);
    _dims.push_back(j);
    _dims.push_back(k);
    _dims.push_back(l);
    _data.resize(i * j * k * l);
}

void Tensor::fill(Type value) noexcept
{
    std::fill(begin(), end(), value);
}

void Tensor::flatten()
{
    PT_ASSERT(isValid());

    auto size = getSize();
    _dims.clear();
    _dims.push_back(size);
}

void Tensor::unpack(std::size_t row, Tensor& out) const
{
    PT_ASSERT(isValid());
    PT_ASSERT(_dims.size() >= 2);
    PT_ASSERT(row < _dims[0]);

    auto packSize = std::accumulate(_dims.begin() + 1, _dims.end(), std::size_t(0));
    auto base = row * packSize;
    auto first = begin() + long(base);
    auto last = first + long(packSize);

    out._dims.clear();
    out._dims.reserve(_dims.size() - 1);
    out._dims.insert(out._dims.end(), _dims.begin() + 1, _dims.end());

    out._data.clear();
    out._data.reserve(std::size_t(last - first));
    out._data.insert(out._data.end(), first, last);
}

void Tensor::select(std::size_t row, Tensor& out) const
{
    unpack(row, out);
    out._dims.insert(out._dims.begin(), 1);
}

void Tensor::add(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const
{
    PT_ASSERT(_dims == other._dims);

    int threads = int(dispatcher.threads());
    int threadSize = int(getSize()) / threads;
    copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        addImpl<Vector2Add>(other, out, dispatcher);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        addImpl<VectorAdd>(other, out, dispatcher);
    }
    else
    {
        addImpl<ScalarAdd>(other, out, dispatcher);
    }
}

void Tensor::multiply(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const
{
    PT_ASSERT(isValid());
    PT_ASSERT(_dims == other._dims);

    int threads = int(dispatcher.threads());
    int threadSize = int(getSize()) / threads;
    copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        multiplyImpl<Vector2Multiply>(other, out, dispatcher);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        multiplyImpl<VectorMultiply>(other, out, dispatcher);
    }
    else
    {
        multiplyImpl<ScalarMultiply>(other, out, dispatcher);
    }
}

void Tensor::dot(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const
{
    PT_ASSERT(_dims.size() == 2);
    PT_ASSERT(other._dims.size() == 2);
    PT_ASSERT(_dims[1] == other._dims[1]);

    out.resize(_dims[0], other._dims[0]);

    int threads = int(dispatcher.threads());
    int threadSize = int(_dims[1]) / threads;

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        dotImpl<Vector2MultiplyAdd>(*this, other, out, dispatcher);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        dotImpl<VectorMultiplyAdd>(*this, other, out, dispatcher);
    }
    else
    {
        dotImpl<ScalarMultiplyAdd>(*this, other, out, dispatcher);
    }
}

void Tensor::fma(const Tensor& scale, const Tensor& bias, Tensor& out, Dispatcher& dispatcher) const
{
    PT_ASSERT(_dims == scale._dims);
    PT_ASSERT(_dims == bias._dims);

    int threads = int(dispatcher.threads());
    int threadSize = int(getSize()) / threads;
    bias.copyTo(out);

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        multiplyAddImpl<Vector2MultiplyAdd>(scale, *this, out, dispatcher);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        multiplyAddImpl<VectorMultiplyAdd>(scale, *this, out, dispatcher);
    }
    else
    {
        multiplyAddImpl<ScalarMultiplyAdd>(scale, *this, out, dispatcher);
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
                --index;
                --numDims;
            }
        }
    }
}

void Tensor::clear() noexcept
{
    _dims.clear();
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
