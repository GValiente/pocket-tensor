#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include "catch.hpp"
#include "pt_tensor.h"

void testModel(pt::Tensor& in, const pt::Tensor& expected, const char* modelFileName, float eps);

#endif
