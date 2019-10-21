/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_TENSOR_H
#define PT_TENSOR_H

#include <vector>
#include <memory>
#include <iosfwd>
#include "pt_libsimdpp.h"
#include "pt_assert.h"

namespace pt
{

class Dispatcher;

class Tensor
{

public:
    using Type = FloatType;
    using Vector = FloatVector;
    static constexpr auto VectorSize = FloatSize;
    static constexpr auto Alignment = sizeof(Type) * VectorSize;

    using DimsVector = std::vector<std::size_t>;
    using DataVector = std::vector<Type, simdpp::aligned_allocator<Type, Alignment>>;

    static std::unique_ptr<Tensor> create(std::size_t dims, std::istream& stream);

    Tensor() = default;

    Tensor(std::size_t i)
    {
        resize(i);
    }

    Tensor(std::size_t i, std::size_t j)
    {
        resize(i, j);
    }

    Tensor(std::size_t i, std::size_t j, std::size_t k)
    {
        resize(i, j, k);
    }

    Tensor(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
    {
        resize(i, j, k, l);
    }

    bool isValid() const noexcept
    {
        return ! _dims.empty();
    }

    const DimsVector& getDims() const noexcept
    {
        return _dims;
    }

    std::size_t getSize() const noexcept;

    const DataVector& getData() const noexcept
    {
        return _data;
    }

    Type operator()(std::size_t i) const noexcept
    {
        return const_cast<Tensor&>(*this).operator()(i);
    }

    Type& operator()(std::size_t i) noexcept
    {
        PT_ASSERT(_dims.size() == 1);
        PT_ASSERT(i < _dims[0]);

        return _data[i];
    }

    Type operator()(std::size_t i, std::size_t j) const noexcept
    {
        return const_cast<Tensor&>(*this).operator()(i, j);
    }

    Type& operator()(std::size_t i, std::size_t j) noexcept
    {
        PT_ASSERT(_dims.size() == 2);
        PT_ASSERT(i < _dims[0]);
        PT_ASSERT(j < _dims[1]);

        return _data[_dims[1] * i + j];
    }

    Type operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept
    {
        return const_cast<Tensor&>(*this).operator()(i, j, k);
    }

    Type& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept
    {
        PT_ASSERT(_dims.size() == 3);
        PT_ASSERT(i < _dims[0]);
        PT_ASSERT(j < _dims[1]);
        PT_ASSERT(k < _dims[2]);

        return _data[_dims[2] * (_dims[1] * i + j) + k];
    }

    Type operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const noexcept
    {
        return const_cast<Tensor&>(*this).operator()(i, j, k, l);
    }

    Type& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) noexcept
    {
        PT_ASSERT(_dims.size() == 4);
        PT_ASSERT(i < _dims[0]);
        PT_ASSERT(j < _dims[1]);
        PT_ASSERT(k < _dims[2]);
        PT_ASSERT(l < _dims[3]);

        return _data[_dims[3] * (_dims[2] * (_dims[1] * i + j) + k) + l];
    }

    typename DataVector::const_iterator begin() const noexcept
    {
        return _data.begin();
    }

    typename DataVector::iterator begin() noexcept
    {
        return _data.begin();
    }

    typename DataVector::const_iterator end() const noexcept
    {
        return _data.end();
    }

    typename DataVector::iterator end() noexcept
    {
        return _data.end();
    }

    void copyTo(Tensor& other) const;

    void resize(std::size_t i);

    void resize(std::size_t i, std::size_t j);

    void resize(std::size_t i, std::size_t j, std::size_t k);

    void resize(std::size_t i, std::size_t j, std::size_t k, std::size_t l);

    void setData(DataVector data) noexcept
    {
        PT_ASSERT(_data.size() == data.size());

        _data = std::move(data);
    }

    void fill(Type value) noexcept;

    void flatten();

    void unpack(std::size_t row, Tensor& out) const;

    void repeat(int n, int axis, Tensor& out) const;

    Tensor unpack(std::size_t row) const
    {
        Tensor output;
        unpack(row, output);
        return output;
    }

    void select(std::size_t row, Tensor& out) const;

    Tensor select(std::size_t row) const
    {
        Tensor output;
        select(row, output);
        return output;
    }

    void add(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const;

    Tensor add(const Tensor& other, Dispatcher& dispatcher) const
    {
        Tensor output;
        add(other, output, dispatcher);
        return output;
    }

    void multiply(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const;

    Tensor multiply(const Tensor& other, Dispatcher& dispatcher) const
    {
        Tensor output;
        multiply(other, output, dispatcher);
        return output;
    }

    void dot(const Tensor& other, Tensor& out, Dispatcher& dispatcher) const;

    Tensor dot(const Tensor& other, Dispatcher& dispatcher) const
    {
        Tensor output;
        dot(other, output, dispatcher);
        return output;
    }

    void fma(const Tensor& scale, const Tensor& bias, Tensor& out, Dispatcher& dispatcher) const;

    Tensor fma(const Tensor& scale, const Tensor& bias, Dispatcher& dispatcher) const
    {
        Tensor output;
        fma(scale, bias, output, dispatcher);
        return output;
    }

    void eraseDummyDims() noexcept;

    void clear() noexcept;

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

protected:
    DimsVector _dims;
    DataVector _data;
};

}

#endif
