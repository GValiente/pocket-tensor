/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <cstddef>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

#define kstringify(x) #x

#define cast(x) static_cast<ptrdiff_t>(x)

#define check(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, kstringify(x)); \
        return false; \
    }

#define check_eq(x, y, eps) \
    { \
        auto x_ = static_cast<double>(x); \
        auto y_ = static_cast<double>(y); \
        if (std::abs(x_ - y_) > eps) { \
            printf( \
                "ASSERT [%s:%d] %f isn't equal to %f ('%s' != '%s')\n", \
                __FILE__, __LINE__, x_, y_, kstringify(x), kstringify(y)); \
            return false; \
        } \
    }

#ifdef DEBUG
#define kassert(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
        exit(-1); \
    }
#else
#define kassert(x) ;
#endif

namespace keras {

#define timeit(t, action) \
    { \
        auto begin = std::chrono::high_resolution_clock::now(); \
        check(action); \
        auto end = std::chrono::high_resolution_clock::now(); \
        t = std::chrono::duration<double>(end - begin).count(); \
    }

bool read_uint(std::ifstream& file, unsigned& i) noexcept;
bool read_float(std::ifstream& file, float& f) noexcept;
bool read_floats(std::ifstream& file, float* f, size_t n) noexcept;

} // namespace keras
