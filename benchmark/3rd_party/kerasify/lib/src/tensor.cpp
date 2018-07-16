/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/tensor.h"

namespace keras {

void Tensor::resize(size_t i) noexcept
{
    dims_ = {i};
    data_.resize(i);
}

void Tensor::resize(size_t i, size_t j) noexcept
{
    dims_ = {i, j};
    data_.resize(i * j);
}

void Tensor::resize(size_t i, size_t j, size_t k) noexcept
{
    dims_ = {i, j, k};
    data_.resize(i * j * k);
}

void Tensor::resize(size_t i, size_t j, size_t k, size_t l) noexcept
{
    dims_ = {i, j, k, l};
    data_.resize(i * j * k * l);
}

Tensor Tensor::unpack(size_t row) const noexcept
{
    kassert(dims_.size() >= 2);
    size_t pack_size = std::accumulate(dims_.begin() + 1, dims_.end(), 0u);

    auto base = row * pack_size;
    auto first = begin() + cast(base);
    auto last = begin() + cast(base + pack_size);

    Tensor x;
    x.dims_ = std::vector<size_t>(dims_.begin() + 1, dims_.end());
    x.data_ = std::vector<float>(first, last);
    return x;
}

Tensor Tensor::select(size_t row) const noexcept
{
    auto x = unpack(row);
    x.dims_.insert(x.dims_.begin(), 1);
    return x;
}

Tensor& Tensor::operator+=(const Tensor& other) noexcept
{
    kassert(dims_ == other.dims_);
    std::transform(begin(), end(), other.begin(), begin(), std::plus<>());
    return *this;
}

Tensor Tensor::fma(const Tensor& scale, const Tensor& bias) const noexcept
{
    kassert(dims_ == scale.dims_);
    kassert(dims_ == bias.dims_);

    Tensor result;
    result.dims_ = dims_;
    result.data_.resize(data_.size());

    auto k_ = scale.begin();
    auto b_ = bias.begin();
    auto r_ = result.begin();
    for (auto x_ = begin(); x_ != end();)
        *(r_++) = *(x_++) * *(k_++) + *(b_++);

    return result;
}

Tensor Tensor::multiply(const Tensor& other) const noexcept
{
    kassert(dims_ == other.dims_);

    Tensor result;
    result.dims_ = dims_;
    result.data_.reserve(data_.size());

    std::transform(
        begin(), end(), other.begin(), std::back_inserter(result.data_),
        std::multiplies<>());
    return result;
}

Tensor Tensor::dot(const Tensor& other) const noexcept
{
    kassert(dims_.size() == 2);
    kassert(other.dims_.size() == 2);
    kassert(dims_[1] == other.dims_[1]);

    Tensor tmp{dims_[0], other.dims_[0]};

    auto ts = cast(tmp.dims_[1]);
    auto is = cast(dims_[1]);

    auto i_ = begin();
    for (auto t0 = tmp.begin(); t0 != tmp.end(); t0 += ts, i_ += is) {
        auto o_ = other.begin();
        for (auto t1 = t0; t1 != t0 + ts; ++t1, o_ += is)
            *t1 = std::inner_product(i_, i_ + is, o_, 0.f);
    }
    return tmp;
}

void Tensor::print() const noexcept
{
    std::vector<size_t> steps(dims_.size());
    std::partial_sum(
        dims_.rbegin(), dims_.rend(), steps.rbegin(), std::multiplies<>());

    size_t count = 0;
    for (auto&& it : data_) {
        for (auto step : steps)
            if (count % step == 0)
                printf("[");
        printf("%f", static_cast<double>(it));
        ++count;
        for (auto step : steps)
            if (count % step == 0)
                printf("]");
        if (count != steps[0])
            printf(", ");
    }
    printf("\n");
}

void Tensor::print_shape() const noexcept
{
    printf("(");
    size_t count = 0;
    for (auto&& dim : dims_) {
        printf("%zu", dim);
        if ((++count) != dims_.size())
            printf(", ");
    }
    printf(")\n");
}

bool Tensor::load(std::ifstream& file, size_t dims) noexcept
{
    check(dims > 0);

    dims_.reserve(dims);
    for (size_t i = 0; i < dims; ++i) {
        unsigned stride = 0;
        check(read_uint(file, stride));
        check(stride > 0);
        dims_.push_back(stride);
    }
    const auto items = size();
    data_.resize(items);
    check(read_floats(file, data_.data(), items));

    return true;
}

} // namespace keras
