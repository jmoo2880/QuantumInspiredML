//
// Created by Zakhar on 16/03/2017.
//

#pragma once

#include "../MathIO/MatrixReadWrite.h"
#include "../Algebra/Matrix.h"
#include "../Algebra/MissingBlock.hpp"

namespace Performance
{

std::vector<int64_t> *benchmarkCDtime(Algebra::Matrix *mx, uint64_t istep);

struct ResultActionDecomposition
{
    Algebra::Matrix Load;
    Algebra::Matrix Rel;
    std::vector<double> CentroidValues;
    std::vector<int64_t> Runtime;
    std::vector<double> Precision;
    
    ResultActionDecomposition()
            : Load(Algebra::Matrix::empty()),
              Rel(Algebra::Matrix::empty())
    { }
};

ResultActionDecomposition
Decomposition(MathIO::MatrixReader &reader, Algebra::Matrix &mat, uint64_t truncation,
              uint64_t istep, uint64_t max, std::string &signVector);

std::vector<int64_t>
Recovery(MathIO::MatrixReader &reader, Algebra::Matrix &mat,
         uint64_t truncation, uint64_t istep, uint64_t max,
         std::string &signVector,
         bool useBatch, bool useNormalization);

int64_t Normalization(Algebra::Matrix &mat);

} // namespace Performance
