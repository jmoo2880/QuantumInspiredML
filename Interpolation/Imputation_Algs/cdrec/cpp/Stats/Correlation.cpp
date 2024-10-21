//
// Created by zakhar on 13.03.18.
//

#include <cmath>

#include "Correlation.h"
#include "../Algebra/SVDecomposition.h"

using namespace Algebra;

namespace Stats
{
//
// Constructors & destructors
//

CorrelationMatrix::CorrelationMatrix(Algebra::Matrix &mx)
        : matrix(mx),
          mean(std::vector<double>(matrix.dimM())),
          stddev(std::vector<double>(matrix.dimM()))
{
    correlation = Matrix::emptyPtr();
    *correlation = Matrix::identity(mx.dimM());
}

CorrelationMatrix::~CorrelationMatrix()
{
    delete correlation;
}

//
// API
//
Algebra::Matrix &Stats::CorrelationMatrix::getCorrelationMatrix()
{
    // step 1 - mean and stddev
    setMeanAndStdDev();
    
    // step 2 - calculate covariance
    for (uint64_t j1 = 0; j1 < matrix.dimM() - 1; ++j1)
    {
        for (uint64_t j2 = j1 + 1; j2 < matrix.dimM(); ++j2)
        {
            (*correlation)(j1, j2) = (*correlation)(j2, j1) = getCorrelation(j1, j2);
        }
    }
    
    corrCalculated = true;
    
    return *correlation;
}

Algebra::Vector Stats::CorrelationMatrix::getSingularValuesOfCM()
{
    if (!corrCalculated)
    {
        getCorrelationMatrix();
    }
    
    Vector sigma(correlation->dimN(), false);
    Matrix svd_v(correlation->dimM(), correlation->dimM(), false);
    
    Algorithms::SVDecomposition::SVDecompose(*correlation, sigma, svd_v);
    
    uint64_t n = sigma.dim();
    
    for (uint64_t shell = n / 2; shell > 0; shell /= 2)
    {
        for (uint64_t i = shell; i < n; ++i)
        {
            double temp = sigma[i];
            uint64_t j;
            for (j = i; j >= shell && sigma[j - shell] < temp; j -= shell)
            {
                sigma[j] = sigma[j - shell];
            }
            sigma[j] = temp;
        }
    }
    
    return sigma;
}


void CorrelationMatrix::normalizeMatrix()
{
    // step 1 - mean and stddev
    setMeanAndStdDev();
    
    // step 2 - normalize
    for (uint64_t i = 0; i < matrix.dimN(); ++i)
    {
        for (uint64_t j = 0; j < matrix.dimM(); ++j)
        {
            matrix(i, j) = (matrix(i, j) - mean[j]) / stddev[j];
        }
    }
}


void CorrelationMatrix::deNormalizeMatrix()
{
    for (uint64_t i = 0; i < matrix.dimN(); ++i)
    {
        for (uint64_t j = 0; j < matrix.dimM(); ++j)
        {
            matrix(i, j) = (matrix(i, j) * stddev[j]) + mean[j];
        }
    }
}

//
// Algorithm
//

void Stats::CorrelationMatrix::setMeanAndStdDev()
{
    std::vector<double> shift = std::vector<double>(matrix.dimM());
    
    for (uint64_t j = 0; j < matrix.dimM(); ++j)
    {
        mean[j] = 0.0;
        stddev[j] = 0.0;
        
        shift[j]  = (matrix(0, j)
                     + matrix(matrix.dimN() - 1, j)
                     + matrix((matrix.dimN() - 1) / 2, j)
                    ) / 3;
    }
    
    for (uint64_t i = 0; i < matrix.dimN(); ++i)
    {
        for (uint64_t j = 0; j < matrix.dimM(); ++j)
        {
            double entry = matrix(i, j) - shift[j];
            mean[j] += entry;
            stddev[j] += entry * entry;
        }
    }
    
    for (uint64_t j = 0; j < matrix.dimM(); ++j)
    {
        stddev[j] -= (mean[j] * mean[j]) / (double)matrix.dimN();
        stddev[j] /= (double)(matrix.dimN() - 1);
        stddev[j] = std::sqrt(stddev[j]);
    
        mean[j] /= (double)matrix.dimN();
        mean[j] += shift[j];
    }
}

double Stats::CorrelationMatrix::getCorrelation(uint64_t col1, uint64_t col2)
{
    double mean1 = mean[col1];
    double mean2 = mean[col2];
    
    double stddev1 = stddev[col1];
    double stddev2 = stddev[col2];
    
    double cov = 0;
    
    for (uint64_t i = 0; i < matrix.dimN(); ++i)
    {
        double val1 = matrix(i, col1);
        double val2 = matrix(i, col2);
        
        cov += ((val1 - mean1) / stddev1) * ((val2 - mean2) / stddev2);
    }
    
    return cov / (double)(matrix.dimN() - 1);
}

const std::vector<double> &CorrelationMatrix::getMean() const
{
    return mean;
}

const std::vector<double> &CorrelationMatrix::getStddev() const
{
    return stddev;
}


} // namespace Stats



