//
// Created by zakhar on 13.03.18.
//

#include "../Algebra/Matrix.h"

#pragma once

namespace Stats
{

class CorrelationMatrix
{
    //
    // Data
    //
  private:
    Algebra::Matrix &matrix;
    Algebra::Matrix *correlation;
    
    std::vector<double> mean;
    std::vector<double> stddev;
    
    bool corrCalculated = false;
    
    //
    // Constructors & destructors
    //
  public:
    explicit CorrelationMatrix(Algebra::Matrix &mx);
    
    ~CorrelationMatrix();
    
    //
    // API
    //
  public:
    Algebra::Matrix &getCorrelationMatrix();
    
    Algebra::Vector getSingularValuesOfCM();
    
    void normalizeMatrix();
    
    void deNormalizeMatrix();
    
    const std::vector<double> &getMean() const;
    const std::vector<double> &getStddev() const;
    //
    // Algorithm
    //
  private:
    void setMeanAndStdDev();
    
    double getCorrelation(uint64_t col1, uint64_t col2);
};

} // namespace Stats
