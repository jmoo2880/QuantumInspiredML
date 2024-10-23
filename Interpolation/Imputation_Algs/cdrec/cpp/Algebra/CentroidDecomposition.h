//
// Created by Zakhar on 08.03.2017.
//

#include "Matrix.h"

#pragma once

namespace Algorithms
{

enum class CDSignVectorStrategy
{
    ISSVBase = 1,
    ISSVPlusBase = 2,
    ISSVInit = 11,
    ISSVPlusInit = 12,
    LSVBase = 21,
    LSVNoInit = 22
};

bool isValidStrategy(CDSignVectorStrategy _strategy);

class CentroidDecomposition
{
    //
    // Data
    //
  private:
    Algebra::Matrix &Src;
    Algebra::Matrix Load;
    Algebra::Matrix Rel;
  
  public:
    std::vector<Algebra::Vector> signVectors;
    //std::vector<std::vector<Algebra::Vector *>> signVectorSteps;
    uint64_t ssvIterations = 0;
    uint64_t truncation = 0;
    
    CDSignVectorStrategy strategy;
    
    //
    // Constructors & destructors
    //
  public:
    explicit CentroidDecomposition(Algebra::Matrix &mx);
    
    CentroidDecomposition(Algebra::Matrix &mx, uint64_t k);
    
    ~CentroidDecomposition() = default;
    
    //
    // API
    //
  public:
    const Algebra::Matrix &getLoad();
    
    const Algebra::Matrix &getRel();
    
    Algebra::Matrix stealLoad();
    
    Algebra::Matrix stealRel();
    
    void destroyDecomposition();
    
    void resetSignVectors();
    
    void performDecomposition(std::vector<double> *centroidValues = nullptr,
                              bool stopOnIncompleteRank = false, bool skipSSV = false);
    
    void increment(const Algebra::Vector &vec);
    
    void increment(const std::vector<double> &vec);
    
    //
    // Algorithm
    //
  private:
    bool decomposed = false;
    uint64_t addedRows = 0;
    
    Algebra::Vector &findSignVector(Algebra::Matrix &mx, uint64_t k);
    
    Algebra::Vector &findLocalSignVector(Algebra::Matrix &mx, uint64_t k, bool useInit);
    
    Algebra::Vector &findOptimizedSSV(Algebra::Matrix &mx, uint64_t k);
    
    Algebra::Vector &findIncrementalSSV(Algebra::Matrix &mx, uint64_t k);
    
    Algebra::Vector &findIncrementalSSVPlus(Algebra::Matrix &mx, uint64_t k);
  
    //
    // Static
    //
  public:
    static constexpr double eps = 1E-11;
    
    static constexpr uint64_t minusone = static_cast<uint64_t>(-1);
    
    static constexpr CDSignVectorStrategy defaultSignVectorStrategy = CDSignVectorStrategy::LSVBase;
    
    static std::pair<Algebra::Matrix, Algebra::Matrix> PerformCentroidDecomposition(Algebra::Matrix &mx, uint64_t k = 0);
};

} // namespace Algorithms