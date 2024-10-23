//
// Created by zakhar on 01/11/18.
//

#pragma once

#include <cstdlib>
#include <tuple>

extern "C"
{
void
centroidDecomposition(
        double *matrixNative, size_t dimN, size_t dimM,
        double *loadContainer, double *relContainer
);

void
centroidDecompositionTruncated(
        double *matrixNative, size_t dimN, size_t dimM,
        double *loadContainer, double *relContainer,
        size_t truncation
);


void
recoveryOfMissingValues(
        double *matrixNative, size_t dimN, size_t dimM,
        size_t truncation, double epsilon
);

void
recoveryOfMissingValuesParametrized(
        double *matrixNative, size_t dimN, size_t dimM,
        size_t truncation, double epsilon,
        size_t useNormalization, size_t optimization,
        size_t signVectorStrategyCode
);

}