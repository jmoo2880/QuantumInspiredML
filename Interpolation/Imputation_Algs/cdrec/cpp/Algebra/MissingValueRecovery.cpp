//
// Created by zakhar on 05.03.18.
//

#include "MissingValueRecovery.h"
#include "../Stats/Correlation.h"

#include <iostream>

using namespace Algebra;

namespace Algorithms
{

MissingValueRecovery::MissingValueRecovery(Algebra::Matrix &src,
                                           uint64_t maxIterations,
                                           double eps)
        : matrix(src),
          cd(src, src.dimM() - 1),
          k(src.dimM() - 1),
          cm(src),
          maxIterations(maxIterations), //defaults are hardcoded in MVR.h
          epsPrecision(eps),
          missingBlocks(std::vector<MissingBlock>()),
          disableCaching(false)
{ }

uint64_t MissingValueRecovery::getReduction()
{
    return k;
}

void MissingValueRecovery::setReduction(uint64_t k)
{
    this->k = k;
    cd.truncation = k;
}

void MissingValueRecovery::passSignVectorStrategy(CDSignVectorStrategy strategy)
{
    cd.strategy = strategy;
}

void MissingValueRecovery::addMissingBlock(uint64_t col, uint64_t start, uint64_t size)
{
    missingBlocks.emplace_back(MissingBlock(col, start, size, matrix));
}

void MissingValueRecovery::addMissingBlock(MissingBlock mb)
{
    missingBlocks.emplace_back(mb);
}

void MissingValueRecovery::autoDetectMissingBlocks(double val)
{
    for (uint64_t j = 0; j < matrix.dimM(); ++j)
    {
        bool missingBlock = false;
        uint64_t start = 0;
        
        for (uint64_t i = 0; i < matrix.dimN(); ++i)
        {
            if ((std::isnan(val) && std::isnan(matrix(i, j))) || (!std::isnan(val) && matrix(i, j) == val))
            {
                if (!missingBlock)
                {
                    missingBlock = true;
                    start = i;
                }
            }
            else
            {
                if (missingBlock)
                {
                    //finalize block
                    missingBlock = false;
                    addMissingBlock(j, start, i - start);
                }
            }
        }
        
        if (missingBlock)
        {
            addMissingBlock(j, start, matrix.dimN() - start);
        }
    }
}

//
// Algorithm
//
void MissingValueRecovery::decomposeOnly()
{
    this->cd.performDecomposition(nullptr);
}

void MissingValueRecovery::increment(const std::vector<double> &vec)
{
    cd.increment(vec);
    
    if (useNormalization && false)
    {
        uint64_t lastIdx = matrix.dimN() - 1;
        const std::vector<double> &mean = cm.getMean();
        const std::vector<double> &stddev = cm.getStddev();
        
        for (uint64_t j = 0; j < matrix.dimM(); ++j)
        {
            matrix(lastIdx, j) = (matrix(lastIdx, j) - mean[j]) / stddev[j];
        }
    }
}

void MissingValueRecovery::increment(const Algebra::Vector &vec)
{
    cd.increment(vec);
    
    if (useNormalization && false)
    {
        uint64_t lastIdx = matrix.dimN() - 1;
        const std::vector<double> &mean = cm.getMean();
        const std::vector<double> &stddev = cm.getStddev();
        
        for (uint64_t j = 0; j < matrix.dimM(); ++j)
        {
            matrix(lastIdx, j) = (matrix(lastIdx, j) - mean[j]) / stddev[j];
        }
    }
}

#define RECOVERY_VERBOSE_
#define determine_reduction_nonstat

uint64_t MissingValueRecovery::performRecovery(bool determineReduction /*= false*/)
{
    uint64_t totalMBSize = 0;
    
    for (auto mblock : missingBlocks)
    {
        totalMBSize += mblock.blockSize;
    }
    
    interpolate();
    
    uint64_t iter = 0;
    double delta = 99.0;
    
    if (useNormalization)
    {
        cm.normalizeMatrix();
    }
    
    if (determineReduction)
    {
        this->determineReduction();
    }
    
    auto centroidValues = std::vector<double>();
    
    while (++iter <= maxIterations && delta >= epsPrecision)
    {
        if (optimization > 0)
        {
            if (iter != 1)
            {
                std::cout << "iteration #" << iter << ", delta=" << delta << std::endl;
            }
            
            bool skipSSV;
            
            switch (optimization)
            {
                case 100:
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                    skipSSV = iter >= (optimization + 1);
                    if (!skipSSV) cd.resetSignVectors();
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 101:
                case 102:
                case 103:
                case 104:
                case 105:
                case 106:
                case 107:
                    optimization -= 100;
                    skipSSV = iter >= (optimization + 1);
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 10:
                    skipSSV = delta <= 1E-2;
                    if (!skipSSV) cd.resetSignVectors();
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 110:
                    skipSSV = delta <= 1E-2;
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 20:
                    skipSSV = iter != 1 && iter != 7;
                    if (!skipSSV) cd.resetSignVectors();
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 120:
                    skipSSV = iter != 1 && iter != 7;
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 25:
                    skipSSV = iter > 3 && iter % 2 == 0; //skips 4, 6, 8 ...
                    if (!skipSSV) cd.resetSignVectors();
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 125:
                    skipSSV = iter > 3 && iter % 2 == 0;
                    this->cd.performDecomposition(&centroidValues, false, skipSSV);
                    break;
                
                case 301:
                case 302:
                case 303:
                case 304:
                case 305:
                case 306:
                case 307:
                    optimization -= 300;
                    if (iter < (optimization + 1))
                    { cd.resetSignVectors(); }
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                case 310:
                    if (delta > 1E-2)
                    { cd.resetSignVectors(); }
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                case 325:
                    if (iter % 2 == 0)
                    { cd.resetSignVectors(); }
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                case 330:
                    if (iter == 1 || iter == 5)
                    { cd.resetSignVectors(); }
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                case 331:
                    if (iter == 1 || iter == 4 || iter == 7)
                    { cd.resetSignVectors(); }
                    this->cd.performDecomposition(&centroidValues, false, false);
                    break;
                
                default:
                    std::cout << "Unrecognized optimization type " << optimization << " for RCD";
                    exit(-1);
            }
        }
        else
        {
            if (disableCaching)
            { cd.resetSignVectors(); }
            
            this->cd.performDecomposition(&centroidValues);
        }

#ifdef RECOVERY_VERBOSE
        std::cout << "CValues: ";
        for (auto a : centroidValues)
        {
            std::cout << a << " ";
        }
        std::cout << std::endl << std::endl;
#endif
        
        const Matrix &L = cd.getLoad();
        const Matrix &R = cd.getRel();
        
        Matrix recover = matrix_mult_A_BT(L, R);
        
        delta = 0.0;
        
        for (auto mblock : missingBlocks)
        {
            for (uint64_t i = mblock.startingIndex; i < mblock.startingIndex + mblock.blockSize; ++i)
            {
                double diff = matrix(i, mblock.column) - recover(i, mblock.column);
                delta += fabs(diff);
                
                matrix(i, mblock.column) = recover(i, mblock.column);
            }
            
            Vector v = mblock.extractBlock();
#ifdef RECOVERY_VERBOSE
            std::cout << "MB_" << iter << " =" << v.toString() << std::endl;
#endif
        }
        
        delta = delta / (double)totalMBSize;
        
        //std::cout << "X_" << iter << " =" << std::endl << matrix->toString() << std::endl;
#ifdef RECOVERY_VERBOSE
        std::cout << "delta_" << iter << " =" << delta << std::endl << std::endl;
#endif
    }
    
    if (useNormalization)
    {
        cm.deNormalizeMatrix();
    }
    
    //std::cout << "X_fin =" << std::endl << matrix->toString() << std::endl;

#ifdef RECOVERY_VERBOSE
    for (auto mblock : missingBlocks) {
        Vector v = Vector(mblock.blockSize, mblock.extractBlock(), true);

        std::cout << "MB_recovery" << " =" << v.toString() << std::endl;
    }
#endif
    
    std::cout << "recovery performed in " << iter - 1 << " iterations " << std::endl;
    
    // when the recovery is done, we need to clean up some stuff
    missingBlocks.clear();
    
    return iter - 1;
}

void MissingValueRecovery::interpolate()
{
    // init missing blocks
    for (auto mblock : missingBlocks)
    {
        // linear interpolation
        double val1 = NAN, val2 = NAN;
        if (mblock.startingIndex > 0)
        {
            val1 = matrix(mblock.startingIndex - 1, mblock.column);
        }
        if (mblock.startingIndex + mblock.blockSize < matrix.dimN())
        {
            val2 = matrix(mblock.startingIndex + mblock.blockSize, mblock.column);
        }
        
        double step;
        
        // fallback case - no 2nd value for interpolation
        if (std::isnan(val1) && std::isnan(val2))
        {
            // starting conditions violation
            abort();
        }
        else if (std::isnan(val1)) // start block is missing
        {
            val1 = val2;
            step = 0;
        }
        else if (std::isnan(val2)) // end block is missing
        {
            step = 0;
        }
        else
        {
            step = (val2 - val1) / (double)(mblock.blockSize + 1);
        }
        
        for (uint64_t i = 0; i < mblock.blockSize; ++i)
        {
            matrix(mblock.startingIndex + i, mblock.column) = val1 + step * (double)(i + 1);
        }
    }
}

void MissingValueRecovery::determineReduction()
{
#ifdef determine_reduction_nonstat
    // step 1 - do full CD to determine rank
    
    std::vector<double> centroidValues = std::vector<double>();
    centroidValues.reserve(matrix.dimM());
    cd.truncation = matrix.dimM();
    cd.performDecomposition(&centroidValues, true);
    
    uint64_t rank = centroidValues.size();
    
    double squaresum = 0.0;
    
    std::cout << "CValues (rank=" << rank << "): ";
    for (auto &a : centroidValues)
    {
        a /= (double)matrix.dimN();
        squaresum += a * a;
        std::cout << a << " ";
    }
    std::cout << std::endl;
    
    // step 2 [ALT] - entropy
    
    std::vector<double> relContribution = std::vector<double>();
    relContribution.reserve(rank);
    for (auto a : centroidValues)
    {
        relContribution.emplace_back(a * a / squaresum);
    }
    
    double entropy = 0.0;
    for (auto a : relContribution)
    {
        entropy += a * std::log(a);
    }
    entropy /= -std::log(rank);
    
    uint64_t red;
    double contributionSum = relContribution[0];
    for (red = 1; red < rank - 1; ++red)
    {
        if (contributionSum >= entropy)
        { break; }
        contributionSum += relContribution[red];
    }
    
    std::cout << "Auto-reduction [entropy] detected as: "
              << red << " in [1..." << rank - 1 << "]," << std::endl
              << "with  sum(contrib)=" << contributionSum
              << " entropy=" << entropy << std::endl << std::endl;
    
    // cleanup - we will have less dimensions later
    cd.destroyDecomposition();
#else
    
    Stats::CorrelationMatrix cm(matrix);

        //cm.normalizeMatrix();

        Matrix *cormat = cm.getCorrelationMatrix();

        std::cout << "Corr(X) =" << std::endl << cormat->toString() << std::endl;

        Vector *sigma = cm.getSingularValuesOfCM();

        std::cout << "Sigma(Corr(X)) =" << sigma->toString() << std::endl;

        uint64_t rank = 0;
        double squaresum = 0.0;

        for (uint64_t i = 0; i < sigma->_dim(); ++i)
        {
            if ((*sigma)[i] < CentroidDecomposition::eps)
            {
                rank = i;
                break;
            }
            else
            {
                squaresum += (*sigma)[i] * (*sigma)[i];
            }
        }
        if (rank == 0) rank = sigma->_dim();

        std::vector<double> relContribution = std::vector<double>();
        relContribution.reserve(rank);
        for (uint64_t i = 0; i < sigma->_dim(); ++i)
        {
            double a = (*sigma)[i];
            relContribution.emplace_back(a*a / squaresum);
        }

        double entropy = 0.0;
        for (auto a : relContribution)
        {
            entropy += a * std::log(a);
        }
        entropy /= -std::log(rank);

        uint64_t red;
        double contributionSum = relContribution[0];
        for (red = 1; red < rank - 1; ++red)
        {
            if (contributionSum >= entropy) { break; }
            contributionSum += relContribution[red];
        }

        std::cout << "Auto-reduction [entropy] detected as: "
                  << red << " in [1..." << rank-1 << "], with  sum(contrib)=" << contributionSum
                  << " entropy=" << entropy << std::endl << std::endl;
#endif
    
    setReduction(red);
}

void MissingValueRecovery::RecoverMatrix(Algebra::Matrix &matrix, uint64_t k, double eps)
{
    MissingValueRecovery rmv(matrix, 100, eps);
    
    rmv.setReduction(k);
    rmv.autoDetectMissingBlocks();
    rmv.performRecovery(k == 0);
}


} // namespace Algorithms
