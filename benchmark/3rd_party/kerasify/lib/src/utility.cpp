/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/utility.h"
#include <algorithm>
#include <iterator>

namespace keras {

bool read_uint(std::ifstream& file, unsigned& i) noexcept
{
    file.read(reinterpret_cast<char*>(&i), sizeof(unsigned));
    check(file.gcount() == sizeof(unsigned));
    return true;
}

bool read_float(std::ifstream& file, float& f) noexcept
{
    file.read(reinterpret_cast<char*>(&f), sizeof(float));
    check(file.gcount() == sizeof(float));
    return true;
}

bool read_floats(std::ifstream& file, float* f, size_t n) noexcept
{
    check(f);

    auto pos = reinterpret_cast<char*>(f);
    auto size = cast(sizeof(float) * n);

    file.read(pos, size);
    check(file.gcount() == size);
    return true;
}

} // namespace keras
