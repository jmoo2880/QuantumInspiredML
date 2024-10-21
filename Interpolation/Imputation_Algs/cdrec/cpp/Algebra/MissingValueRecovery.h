//
// Created by zakhar on 05.03.18.
//

#pragma once

#include <cmath>

#include "CentroidDecomposition.h"
#include "MissingBlock.hpp"
#include "../Stats/Correlation.h"

namespace Algorithms
{

class MissingValueRecovery
{
    //
    // Data
    //
  private:
    Algebra::Matrix &matrix;
    CentroidDecomposition cd;
    uint64_t k;
    Stats::CorrelationMatrix cm;
  
  public:
    const uint64_t maxIterations;
    double epsPrecision;
    std::vector<MissingBlock> missingBlocks;
    
    uint64_t optimization = 0;
    bool disableCaching = false;
    bool useNormalization = false;
    
    //
    // Constructors & desctructors
    //
  public:
    explicit MissingValueRecovery(Algebra::Matrix &src, uint64_t maxIterations = 100, double eps = 1E-6);
    
    //
    // API
    //
  public:
    uint64_t getReduction();
    
    void setReduction(uint64_t k);
    
    void passSignVectorStrategy(CDSignVectorStrategy strategy);
    
    void addMissingBlock(uint64_t col, uint64_t start, uint64_t size);
    
    void addMissingBlock(MissingBlock mb);
    
    void autoDetectMissingBlocks(double val = NAN);
    
    void decomposeOnly();
    
    void increment(const std::vector<double> &vec);
    
    void increment(const Algebra::Vector &vec);
    
    uint64_t performRecovery(bool determineReduction = false);
    
    //
    // Algorithm
    //
  private:
    void interpolate();
    
    void determineReduction();
    
    //
    // Static
    //
  public:
    static void RecoverMatrix(Algebra::Matrix &matrix, uint64_t k = 0, double eps = 1E-6);
};
} // namespace Algorithms
