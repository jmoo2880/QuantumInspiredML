//
// Created by zakhar on 14.03.18.
//

#pragma once

#include "Matrix.h"

namespace Algebra
{
namespace Algorithms
{

class SVDecomposition
{
  public:
    static int64_t SVDecompose(Algebra::Matrix &u,
                               Algebra::Vector &sigma,
                               Algebra::Matrix &v);
};

} // namespace Algorithms
} // namespace Algebra