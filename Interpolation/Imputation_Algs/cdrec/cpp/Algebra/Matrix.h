//
// Created by Zakhar on 07.03.2017.
//

#pragma once

#include <vector>
#include <string>

#include "Vector.h"
#include "../shared/SharedLibFunctions.h"

namespace Algebra
{

class Matrix
{
    //
    // Data
    //
  private:
    uint64_t _dimNreal;
  
  protected:
    double *_data;
    uint64_t _dimN;
    uint64_t _dimM;
    bool _isReference = false;
  
  public:
    double &operator()(uint64_t i, uint64_t j);
    
    double &operator()(uint64_t i, uint64_t j) const;
    
    uint64_t dimN() const;
    
    uint64_t dimM() const;
    
    /// Creates a copy of the matrix with all its data
    Matrix copy() const;
    
    /// Generates a string representation of the matrix
    std::string toString() const;
  
  private:
    void expandStorage();
    
    //
    // Constructors & destructors
    //
  private:
    explicit Matrix();
  
  public:
    Matrix(uint64_t n, uint64_t m, double *data, bool isReference = false);
    
    Matrix(uint64_t n, uint64_t m, bool init);
    
    explicit Matrix(const std::vector<std::vector<double>> &data);
    
    /// Not copy-constructible
    Matrix(const Matrix &other) = delete;
    
    Matrix(Matrix &other) = delete;
    
    Matrix &operator=(const Matrix &other) = delete;
    
    Matrix &operator=(Matrix &other) = delete;
    
    /// Move constructor
    /// @param [in] other matrix to move from
    Matrix(Matrix &&other) noexcept;
    
    /// Move assignment
    /// @param [in] other matrix to move from
    Matrix &operator=(Matrix &&other) noexcept;
    
    ~Matrix();
    
    //
    // Pre-defined
    //
  public:
    static Matrix identity(uint64_t dim);
    
    static Matrix empty();
    
    static Matrix *emptyPtr();
    
    static Matrix randomMat(uint64_t n, uint64_t m, bool useSeed = false, uint32_t seed = 0);
    
    //
    // Unary operations
    //
  public:
    void destroy();
    
    double normF() const;
    
    Vector extractRowVector(uint64_t i, bool copy);
    
    Vector extractRowVector(uint64_t i) const;
    
    Vector extractColumnVector(uint64_t j) const;
    
    Matrix &insertVectorAtRow(uint64_t i, const Vector &vec);
    
    Matrix &insertVectorAtColumn(uint64_t j, const Vector &vec);
    
    Matrix &append(const std::vector<double> &newData);
    
    Matrix &append(const Vector &newData);
    
    /// Creates a new matrix by taking first upper-left-most `size_n` x `size_m` submatrix of the current one
    /// @param [in] size_n amount of rows to take for a new matrix
    /// @param [in] size_m amount of columns to take for a new matrix
    Matrix subMatrix(uint64_t size_n, uint64_t size_m);
    
    /// Creates a new matrix by taking `size_n` x `size_m` submatrix of the current one
    /// starting at indices `start_n` for rows and `start_m` for columns
    /// @param [in] start_n index of the first row to take for a new matrix
    /// @param [in] start_m index of the first column to take for a new matrix
    /// @param [in] size_n amount of rows to take for a new matrix
    /// @param [in] size_m amount of columns to take for a new matrix
    Matrix subMatrix(uint64_t start_n, uint64_t start_m, uint64_t size_n, uint64_t size_m);
    
    /// Constructs a vector out of a diagonal of the matrix,
    /// calling the function on a non-square matrix generates an exception
    Vector diag();
    
    Matrix &operator+=(const Matrix &mxB);
    
    Matrix &operator-=(const Matrix &mxB);
    
    //
    // Service
    //
  
  public:
    friend void
    ::centroidDecompositionTruncated(
            double *matrixNative, size_t dimN, size_t dimM,
            double *loadContainer, double *relContainer,
            size_t truncation
    );
    
    friend void
    ::recoveryOfMissingValuesParametrized(
            double *matrixNative, size_t dimN, size_t dimM,
            size_t truncation, double epsilon,
            size_t useNormalization, size_t optimization,
            size_t signVectorStrategyCode
    );
}; // end class

//
// Binary operations
//
Matrix operator+(const Matrix &mxA, const Matrix &mxB);

Matrix operator-(const Matrix &mxA, const Matrix &mxB);

Matrix &operator*(Matrix &mx, double scalar);

Matrix &operator*(double scalar, Matrix &mx);

Matrix operator*(const Matrix &mxA, const Matrix &mxB);

Matrix matrix_mult_AT_B(const Matrix &mxA, const Matrix &mxB);

Matrix matrix_mult_A_BT(const Matrix &mxA, const Matrix &mxB);

//
// Inter-type operations
//
Vector operator*(const Matrix &mx, const Vector &vec);

Vector operator^(const Matrix &mx_t, const Vector &vec);

Matrix vector_outer(const Vector &vecA, const Vector &vecB);

}// namespace Algebra
