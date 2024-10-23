//
// Created by zakhar on 05.03.18.
//

#pragma once

#include "Matrix.h"

namespace Algorithms
{

class MissingBlock
{
    //
    // Data
    //
  public:
    const uint64_t column;
    const uint64_t startingIndex;
    const uint64_t blockSize;
    
    const Algebra::Matrix &matrix;
    
    //
    // Constructors & destructors
    //
  public:
    MissingBlock(uint64_t column, uint64_t startingIndex, uint64_t blockSize,
                 Algebra::Matrix &matrix)
            : column(column), startingIndex(startingIndex), blockSize(blockSize),
              matrix(matrix)
    { }
  
  public:
    Algebra::Vector extractBlock()
    {
        Algebra::Vector extraction(blockSize, false);
        
        for (uint64_t i = 0; i < blockSize; ++i)
        {
            extraction[i] = matrix(startingIndex + i, column);
        }
        
        return extraction;
    }
    
    void imputeBlock(const Algebra::Vector &data)
    {
        for (uint64_t i = 0; i < blockSize; ++i)
        {
            matrix(column, startingIndex + i) = data[i];
        }
    }
};

} // namespace Algorithms
