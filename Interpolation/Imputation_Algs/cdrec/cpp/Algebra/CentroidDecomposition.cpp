//
// Created by Zakhar on 08.03.2017.
//

#include <cmath>
#include <iostream>
#include "CentroidDecomposition.h"

using namespace Algebra;

namespace Algorithms
{
//
// CentroidDecomposition constructors & desctructors
//

CentroidDecomposition::CentroidDecomposition(Algebra::Matrix &mx)
        : Src(mx),
          Load(mx.dimN(), mx.dimM(), true),
          Rel(mx.dimM(), mx.dimM(), true),
          signVectors(std::vector<Vector>()),
          truncation(mx.dimM()),
          strategy(defaultSignVectorStrategy)
{
    // i = 0:
    Vector signV(mx.dimN(), false);
    signV.fill(1.0);
    signVectors.emplace_back(std::move(signV));
    
    // i in [1,m[
    for (uint64_t i = 1; i < mx.dimM(); ++i)
    {
        signVectors.emplace_back(signVectors[0].copy());
    }
}

CentroidDecomposition::CentroidDecomposition(Algebra::Matrix &mx, uint64_t k)
        : CentroidDecomposition(mx)
{
    truncation = k;
}

//
// CentroidDecomposition API
//

const Matrix &CentroidDecomposition::getLoad()
{
    return Load;
}

const Matrix &CentroidDecomposition::getRel()
{
    return Rel;
}

Matrix CentroidDecomposition::stealLoad()
{
    return std::move(Load);
}

Matrix CentroidDecomposition::stealRel()
{
    return std::move(Rel);
}

void CentroidDecomposition::destroyDecomposition()
{
    Load.destroy();
    Rel.destroy();
}

void CentroidDecomposition::resetSignVectors()
{
    for (uint64_t i = 0; i < Src.dimM(); ++i)
    {
        for (uint64_t j = 0; j < Src.dimN(); j++)
        {
            signVectors[i][j] = 1.0;
        }
    }
}

void CentroidDecomposition::performDecomposition(std::vector<double> *centroidValues,
                                                 bool stopOnIncompleteRank /*= false*/, bool skipSSV /*= false*/)
{
    Matrix X = Src.copy();
    
    for (uint64_t i = 0; i < truncation; i++)
    {
        Vector &Z = skipSSV
                    ? signVectors[i]
                    : findSignVector(X, i);
        
        //std::cout << Z.toString() << std::endl;
        
        // C_*i = X^T * Z
        Vector Rel_i = X ^Z;
        
        // R_*i = C_*i / ||C_*i||
        double centroid = Rel_i.norm2();
        if (centroidValues != nullptr)
        {
            if (stopOnIncompleteRank && centroid < eps) // incomplete rank, don't even proceed with current iteration
            {
                break;
            }
            
            centroidValues->emplace_back(centroid);
        }
        
        Rel_i / centroid;
        
        // R = Append(R, R_*i)
        Rel.insertVectorAtColumn(i, Rel_i);
        
        // L_*i = X * R
        Vector Load_i = X * Rel_i;
        
        // L = Append(L, L_*i)
        Load.insertVectorAtColumn(i, Load_i);
        
        // X := X - L_*i * R_*i^T
        X -= vector_outer(Load_i, Rel_i);
    }
    
    addedRows = 0;
    decomposed = true;
}

//
// CentroidDecomposition algorithm
//

void CentroidDecomposition::increment(const Vector &vec)
{
    Src.append(vec);
    Load.append(vec); // doesn't matter, will be overwritten
    ++addedRows;
    
    for (uint64_t i = 0; i < Src.dimM(); ++i)
    {
        signVectors[i].append(1.0);
    }
}

void CentroidDecomposition::increment(const std::vector<double> &vec)
{
    Src.append(vec);
    Load.append(vec); // ditto as ^
    ++addedRows;
    
    for (uint64_t i = 0; i < Src.dimM(); ++i)
    {
        signVectors[i].append(1.0);
    }
}

//
// Algorithm
//

Vector &CentroidDecomposition::findSignVector(Algebra::Matrix &mx, uint64_t k)
{
    switch (strategy)
    {
        case CDSignVectorStrategy::ISSVBase:
            return findIncrementalSSV(mx, k);
        
        case CDSignVectorStrategy::ISSVPlusBase:
            return findIncrementalSSVPlus(mx, k);
        
        case CDSignVectorStrategy::ISSVInit:
        case CDSignVectorStrategy::ISSVPlusInit:
            return findOptimizedSSV(mx, k);
        
        case CDSignVectorStrategy::LSVBase:
            return findLocalSignVector(mx, k, true);
        
        case CDSignVectorStrategy::LSVNoInit:
            return findLocalSignVector(mx, k, false);
    }
}

Vector &CentroidDecomposition::findLocalSignVector(Algebra::Matrix &mx, uint64_t k, bool useInit)
{
    Vector &Z = signVectors[k]; // get a reference
    Vector direction = Vector::empty();
    
    //
    // First pass - init
    //
    if (!decomposed && useInit)
    {
        direction = Vector(mx.dimM(), false);
        
        for (uint64_t j = 0; j < mx.dimM(); ++j)
        {
            direction[j] = mx(0, j);
        }
        
        for (uint64_t i = 1; i < mx.dimN(); ++i)
        {
            double gradPlus = 0.0;
            double gradMinus = 0.0;
            
            for (uint64_t j = 0; j < mx.dimM(); ++j)
            {
                double localModPlus = direction[j] + mx(i, j);
                gradPlus += localModPlus * localModPlus;
                double localModMinus = direction[j] - mx(i, j);
                gradMinus += localModMinus * localModMinus;
            }
            
            // if keeping +1 as a sign yields a net negative to the
            Z[i] = gradPlus > gradMinus ? 1 : -1;
            
            for (uint64_t j = 0; j < mx.dimM(); ++j)
            {
                direction[j] += Z[i] * mx(i, j);
            }
        }
    }
    else // Alternative first pass - init to {+1}^n
    {
        direction = (mx ^ Z);
    }
    
    //
    // 2+ pass - update to Z
    //
    
    bool flipped;
    double lastNorm = // cache the current value of (||D||_2)^2 to avoid recalcs
            vector_dot(direction, direction) + eps; // eps to avoid "parity flip"
    
    do
    {
        flipped = false;
        
        for (uint64_t i = 0; i < mx.dimN(); ++i)
        {
            double signDouble = Z[i] * 2;
            double gradFlip = 0.0;
            
            for (uint64_t j = 0; j < mx.dimM(); ++j)
            {
                double localMod = direction[j] - signDouble * mx(i, j);
                gradFlip += localMod * localMod;
            }
            
            if (gradFlip > lastNorm) // net positive from flipping
            {
                flipped = true;
                Z[i] *= -1;
                lastNorm = gradFlip + eps;
                
                for (uint64_t j = 0; j < mx.dimM(); ++j)
                {
                    direction[j] -= signDouble * mx(i, j);
                }
            }
        }
    } while (flipped);
    
    return Z;
}

Vector &CentroidDecomposition::findOptimizedSSV(Algebra::Matrix &mx, uint64_t k)
{
    if (!decomposed)
    {
        Vector &Z = signVectors[k]; // get a reference
        
        std::vector<double> direction = std::vector<double>(mx.dimM());
        
        for (uint64_t j = 0; j < mx.dimM(); ++j)
        {
            direction[j] = mx(0, j);
        }
        
        for (uint64_t i = 1; i < mx.dimN(); ++i)
        {
            double gradPlus = 0.0;
            double gradMinus = 0.0;
            
            for (uint64_t j = 0; j < mx.dimM(); ++j)
            {
                double localModPlus = direction[j] + mx(i, j);
                gradPlus += localModPlus * localModPlus;
                double localModMinus = direction[j] - mx(i, j);
                gradMinus += localModMinus * localModMinus;
            }
            
            double sign = gradPlus > gradMinus ? 1 : -1;
            Z[i] = sign;
            
            for (uint64_t j = 0; j < mx.dimM(); ++j)
            {
                direction[j] += sign * mx(i, j);
            }
        }
    }
    
    return strategy == CDSignVectorStrategy::ISSVInit ? findIncrementalSSV(mx, k) : findIncrementalSSVPlus(mx, k);
}

Vector &CentroidDecomposition::findIncrementalSSVPlus(Algebra::Matrix &mx, uint64_t k)
{
    // Scalable Sign Vector
    uint64_t pos = minusone;
    double val = 0.0;
    
    Vector &Z = signVectors[k]; // get a reference
    
    Vector S(Src.dimM(), false);
    Vector V(Src.dimN(), false);
    
    // pre-process rows of X
    std::vector<Vector> x_ = std::vector<Vector>();
    
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        x_.push_back(mx.extractRowVector(i, false));
    }
    
    
    // ITERATION #1
    
    // S = Sum(1:n) { z_i * (X_i* ^ T) }
    
    // i = 0
    {
        for (uint64_t j = 0; j < S.dim(); ++j)
        {
            S[j] = x_[0][j] * Z[0];
        }
    }
    
    for (uint64_t i = 1; i < mx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < S.dim(); ++j)
        {
            S[j] += x_[i][j] * Z[i];
        }
    }
    
    
    // v_i = z_i * (z_i * X_i* * S - X_i* * (X_i*)^T)
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        V[i] = Z[i] * (
                Z[i] * vector_dot(x_[i], S) - vector_dot(x_[i], x_[i])
        );
    }
    
    // Search next element
    
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        if (Z[i] * V[i] < 0)
        {
            if (fabs(V[i]) > val)
            {
                val = fabs(V[i]);
                pos = i;
            }
        }
    }
    
    // ITERATIONS 2+
    
    // main loop
    while (pos != minusone)
    {
        // Search next element to flip
        val = eps;
        pos = minusone;
        
        for (uint64_t i = 0; i < mx.dimN(); ++i)
        {
            if (Z[i] * V[i] < 0)
            {
                if (fabs(V[i]) > val)
                {
                    val = fabs(V[i]);
                    pos = i;
                    
                    // change sign
                    Z[pos] = Z[pos] * (-1.0);
                    
                    double factor = Z[pos] + Z[pos];
                    
                    // Determine V_k+1 from V_k
                    
                    for (uint64_t l = 0; l < mx.dimN(); ++l)
                    {
                        V[l] = V[l] + factor * (l == pos ? 0 : vector_dot(x_[l], x_[pos]));
                    }
                }
            }
        }
        ++ssvIterations;
    }
    
    return Z;
}

Vector &CentroidDecomposition::findIncrementalSSV(Algebra::Matrix &mx, uint64_t k)
{
    // Scalable Sign Vector
    uint64_t pos = minusone;
    double val = 0.0;
    
    Vector &Z = signVectors[k]; // get a reference
    
    Vector S(Src.dimM(), false);
    Vector V(Src.dimN(), false);
    
    // pre-process rows of X
    std::vector<Vector> x_ = std::vector<Vector>();
    
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        x_.push_back(mx.extractRowVector(i, false));
    }
    
    // ITERATION #1
    
    // S = Sum(1:n) { z_i * (X_i* ^ T) }
    
    // i = 0
    {
        for (uint64_t j = 0; j < S.dim(); ++j)
        {
            S[j] = x_[0][j] * Z[0];
        }
    }
    
    for (uint64_t i = 1; i < mx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < S.dim(); ++j)
        {
            S[j] += x_[i][j] * Z[i];
        }
    }
    
    
    
    // v_i = z_i * (z_i * X_i* * S - X_i* * (X_i*)^T)
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        V[i] = Z[i] * (
                Z[i] * vector_dot(x_[i], S) - vector_dot(x_[i], x_[i])
        );
    }
    
    // Search next element
    
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        if (Z[i] * V[i] < 0)
        {
            if (fabs(V[i]) > val)
            {
                val = fabs(V[i]);
                pos = i;
            }
        }
    }

    // ITERATIONS 2+
    
    // main loop
    while (pos != minusone)
    {
        // change sign
        Z[pos] = Z[pos] * (-1.0);
        
        // Determine V_k+1 from V_k
        
        for (uint64_t i = 0; i < mx.dimN(); ++i)
        {
            V[i] += 2 * Z[pos] * (i == pos ? 0 : vector_dot(x_[i], x_[pos]));
        }

        // Search next element to flip
        val = 0.0;
        pos = minusone;
        
        for (uint64_t i = 0; i < mx.dimN(); ++i)
        {
            if (Z[i] * V[i] < 0)
            {
                if (fabs(V[i]) > val)
                {
                    val = fabs(V[i]);
                    pos = i;
                }
            }
        }
        ++ssvIterations;
    }
    
    return Z;
}

std::pair<Algebra::Matrix, Algebra::Matrix>
CentroidDecomposition::PerformCentroidDecomposition(Algebra::Matrix &mx, uint64_t k)
{
    k = k == 0 ? mx.dimM() : k;
    CentroidDecomposition cd(mx);
    
    cd.truncation = k;
    
    cd.performDecomposition(nullptr);
    
    return std::make_pair(cd.stealLoad(), cd.stealRel());
}

// volatile function because there's an external cast from an int involved

bool isValidStrategy(CDSignVectorStrategy _strategy)
{
    size_t code = (size_t)_strategy;
    CDSignVectorStrategy strategy = (CDSignVectorStrategy)code;
    volatile bool valid = false;
    valid = valid;
    
    switch (strategy)
    {
        case CDSignVectorStrategy::ISSVBase:
        case CDSignVectorStrategy::ISSVPlusBase:
        case CDSignVectorStrategy::ISSVInit:
        case CDSignVectorStrategy::ISSVPlusInit:
        case CDSignVectorStrategy::LSVBase:
        case CDSignVectorStrategy::LSVNoInit:
            valid = true;
    }
    
    return valid;
}

} // namespace Algorithms
