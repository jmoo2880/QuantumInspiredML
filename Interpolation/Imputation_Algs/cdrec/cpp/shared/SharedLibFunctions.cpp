//
// Created by zakhar on 01/11/18.
//

#include <cassert>

#include "SharedLibFunctions.h"
#include "../Algebra/MissingValueRecovery.h"

extern "C"
{

void
centroidDecomposition(
        double *matrixNative, size_t dimN, size_t dimM,
        double *loadContainer, double *relContainer
)
{
    centroidDecompositionTruncated(matrixNative, dimN, dimM, loadContainer, relContainer, 0);
}

void
centroidDecompositionTruncated(
        double *matrixNative, size_t dimN, size_t dimM,
        double *loadContainer, double *relContainer,
        size_t truncation
)
{
    truncation = truncation == 0 ? dimM : truncation;
    
    Algebra::Matrix matrix(dimN, dimM, matrixNative, true);
    
    Algebra::Matrix Load = Algebra::Matrix::empty();
    Algebra::Matrix Rel = Algebra::Matrix::empty();
    
    std::tie(Load, Rel) = Algorithms::CentroidDecomposition::PerformCentroidDecomposition(matrix, truncation);
    
    // [!] explicitly copy the output
    std::copy(Load._data, Load._data + (dimN * dimM), loadContainer);
    std::copy(Rel._data, Rel._data + (dimM * dimM), relContainer);
}

void
recoveryOfMissingValues(
        double *matrixNative, size_t dimN, size_t dimM,
        size_t truncation, double epsilon
)
{
    Algebra::Matrix matrix(dimN, dimM, matrixNative, true);
    
    Algorithms::MissingValueRecovery::RecoverMatrix(matrix, truncation, epsilon);
    // [!] input already modified
}

void
recoveryOfMissingValuesParametrized(
        double *matrixNative, size_t dimN, size_t dimM,
        size_t truncation, double epsilon,
        size_t useNormalization, size_t optimization,
        size_t signVectorStrategyCode
)
{
    Algebra::Matrix matrix(dimN, dimM, matrixNative, true);
    
    Algorithms::MissingValueRecovery rmv(matrix);
    
    rmv.setReduction(truncation);
    rmv.epsPrecision = epsilon;
    rmv.useNormalization = useNormalization != 0;
    rmv.optimization = optimization;
    
    if (signVectorStrategyCode > 0)
    {
        Algorithms::CDSignVectorStrategy signVectorStrategy = (Algorithms::CDSignVectorStrategy) signVectorStrategyCode;
        assert(Algorithms::isValidStrategy(signVectorStrategy));
        rmv.passSignVectorStrategy(signVectorStrategy);
    }
    
    rmv.autoDetectMissingBlocks();
    rmv.performRecovery(truncation == 0);
    // [!] input already modified
}

}