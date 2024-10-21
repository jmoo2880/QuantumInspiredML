//
// Created by Zakhar on 03.03.2017.
//

#include <cmath>
#include <cassert>

#include <vector>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <random>

#ifdef multi
#include <omp.h>
#endif

#include "Matrix.h"

namespace Algebra
{

//
// Matrix data
//

double &Matrix::operator()(uint64_t i, uint64_t j)
{
    return _data[_dimM * i + j];
}

double &Matrix::operator()(uint64_t i, uint64_t j) const
{
    return _data[_dimM * i + j];
}

uint64_t Matrix::dimN() const
{ return _dimN; }

uint64_t Matrix::dimM() const
{ return _dimM; }

Matrix Matrix::copy() const
{
    double *newData = (double *)std::malloc(_dimN * _dimM * sizeof(*newData));
    std::copy(_data, _data + (_dimN * _dimM), newData);
    
    return Matrix(_dimN, _dimM, newData);
}

std::string Matrix::toString() const
{
    std::stringstream strbuffer;
    
    strbuffer << std::fixed << std::setprecision(4);
    for (uint64_t i = 0; i < _dimN; i++)
    {
        for (uint64_t j = 0; j < _dimM; j++)
        {
            double elem = operator()(i, j);
            strbuffer << (elem >= 0 ? " " : "") << elem << "\t";
        }
        strbuffer << std::endl;
    }
    
    return strbuffer.str();
}

void Matrix::expandStorage()
{
    uint64_t newDim = _dimN + 1;
    
    if (newDim > _dimNreal)
    {
        // reallocate memory
        _dimNreal = _dimN < 4 ? 4 : static_cast<size_t>(std::round((double)_dimNreal * 1.4));
        _data = (double *)std::realloc(_data, _dimNreal * _dimM * sizeof(*_data));
    }
    
    _dimN = newDim;
}

//
// Matrix constructors & destructors
//

Matrix::Matrix()
        : _dimNreal(0),
          _data(nullptr),
          _dimN(0),
          _dimM(0),
          _isReference(false)
{ }

Matrix::Matrix(uint64_t n, uint64_t m, double *data, bool isReference)
        : _dimNreal(n),
          _data(data),
          _dimN(n),
          _dimM(m),
          _isReference(isReference)
{ }

Matrix::Matrix(uint64_t n, uint64_t m, bool init)
        : _dimNreal(n),
          _dimN(n),
          _dimM(m),
          _isReference(false)
{
    _data = (double *)std::malloc(_dimN * _dimM * sizeof(*_data));
    if (init)
    {
        std::fill(_data, _data + (_dimN * _dimM), 0);
    }
}

Matrix::Matrix(const std::vector<std::vector<double>> &data)
        : _dimNreal(data.size()),
          _dimN(data.size()),
          _dimM(data[0].size()),
          _isReference(false)
{
    _data = (double *)std::malloc(_dimN * _dimM * sizeof(*_data));
    
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        for (uint64_t j = 0; j < _dimM; ++j)
        {
            operator()(i, j) = data[i][j];
        }
    }
}

Matrix::Matrix(Matrix &&other) noexcept
        : _dimNreal(other._dimNreal),
          _data(other._data),
          _dimN(other._dimN),
          _dimM(other._dimM),
          _isReference(other._isReference)
{
    other._dimNreal = 0;
    other._data = nullptr;
    other._dimN = 0;
    other._dimM = 0;
    other._isReference = false;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept
{
    if (this != &other)
    {
        if (_data != nullptr && !_isReference)
        { free(_data); }
        
        _dimNreal = other._dimNreal;
        _data = other._data;
        _dimN = other._dimN;
        _dimM = other._dimM;
        _isReference = other._isReference;
        
        other._dimNreal = 0;
        other._data = nullptr;
        other._dimN = 0;
        other._dimM = 0;
        other._isReference = false;
    }
    
    return *this;
}

Matrix::~Matrix()
{
    if (_data != nullptr && !_isReference)
    { std::free(_data); }
}

//
// Pre-defined matrices
//
Matrix Matrix::identity(uint64_t dim)
{
    Matrix mx(dim, dim, true);
    
    for (uint64_t i = 0; i < dim; ++i)
    {
        mx(i, i) = 1.0;
    }
    
    return mx;
}

Matrix Matrix::empty()
{
    return Matrix();
}

Matrix *Matrix::emptyPtr()
{
    return new Matrix();
}

Matrix Matrix::randomMat(uint64_t n, uint64_t m, bool useSeed, uint32_t seed)
{
    Matrix newMat = Matrix(n, m, false);
    
    std::random_device rd;
    std::mt19937 gen(useSeed ? seed : rd());
    std::uniform_real_distribution<double> randGen(-1.0, 1.0);
    
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < m; ++j)
        {
            newMat(i, j) = randGen(gen);
        }
    }
    
    return newMat;
}

//
// Matrix operations
//

void Matrix::destroy()
{
    std::fill(_data, _data + (_dimN * _dimM), 0);
}

double Matrix::normF() const
{
    double norm = 0.0;
    
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        for (uint64_t j = 0; j < _dimM; ++j)
        {
            norm += operator()(i, j) * operator()(i, j);
        }
    }
    
    return sqrt(norm);
}

Vector Matrix::extractRowVector(uint64_t i, bool copy)
{
    Vector newVector(_dimM, (_data + i * _dimM), copy);
    
    return newVector;
}

Vector Matrix::extractRowVector(uint64_t i) const
{
    Vector newVector(_dimM, (_data + i * _dimM), true);
    
    return newVector;
}

Vector Matrix::extractColumnVector(uint64_t j) const
{
    Vector newVector(_dimN, false);
    
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        newVector[i] = operator()(i, j);
    }
    
    return newVector;
}

Matrix &Matrix::insertVectorAtRow(uint64_t i, const Vector &vec)
{
    assert(vec.dim() == _dimM);
    
    std::copy(vec._data, vec._data + vec.dim(), _data + (i * _dimM));
    
    return *this;
}

Matrix &Matrix::insertVectorAtColumn(uint64_t j, const Vector &vec)
{
    assert(vec.dim() == _dimN);
    
    for (uint64_t i = 0; i < vec.dim(); ++i)
    {
        operator()(i, j) = vec[i];
    }
    
    return *this;
}

Matrix &Matrix::append(const std::vector<double> &newData)
{
    if (_isReference)
    {
        throw std::logic_error("Can't append to referenced matrices");
    }
    expandStorage();
    
    for (uint64_t j = 0; j < _dimM; ++j)
    {
        operator()(_dimN - 1, j) = newData[j];
    }
    
    return *this;
}

Matrix &Matrix::append(const Vector &newData)
{
    if (_isReference)
    {
        throw ReferenceAppendedException("Appending to reference-matrix is not allowed, create a copy() first");
    }
    expandStorage();
    
    insertVectorAtRow(_dimN - 1, newData);
    
    return *this;
}

Matrix Matrix::subMatrix(uint64_t size_n, uint64_t size_m)
{
    assert(size_n <= _dimN);
    assert(size_m <= _dimM);
    
    Matrix newMat(size_n, size_m, false);
    
    if (size_m == _dimM) // exceptional case
    {
        std::copy(_data,
                  _data + (size_n * _dimM),
                  newMat._data);
    }
    else
    {
        // row-wise copy
        for (uint64_t i = 0; i < size_n; ++i)
        {
            std::copy(_data + (i * _dimM),
                      _data + (i * _dimM) + size_m,
                      newMat._data + (i * size_m));
        }
    }
    
    return newMat;
}

Matrix Matrix::subMatrix(uint64_t start_n, uint64_t start_m, uint64_t size_n, uint64_t size_m)
{
    assert(start_n + size_n <= _dimN);
    assert(start_m + size_m <= _dimM);
    
    Matrix newMat(size_n, size_m, false);
    
    if (size_m == _dimM) // exceptional case (=> start_m == 0)
    {
        std::copy(_data + (start_n * _dimM),
                  _data + ((start_n + size_n) * _dimM),
                  newMat._data);
    }
    else
    {
        // row-wise copy
        for (uint64_t i = start_n; i < start_n + size_n; ++i)
        {
            std::copy(_data + (i * _dimM) + start_m,
                      _data + (i * _dimM) + start_m + size_m,
                      newMat._data + ((i - start_n) * size_m));
        }
    }
    
    return newMat;
}

Vector Matrix::diag()
{
    assert(_dimN == _dimM);
    
    Vector newVec(_dimN, false);
    
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        newVec[i] = operator()(i, i);
    }
    
    return newVec;
}

Matrix &Matrix::operator+=(const Matrix &mxB)
{
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        for (uint64_t j = 0; j < _dimM; ++j)
        {
            (*this)(i, j) += mxB(i, j);
        }
    }
    
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &mxB)
{
    for (uint64_t i = 0; i < _dimN; ++i)
    {
        for (uint64_t j = 0; j < _dimM; ++j)
        {
            (*this)(i, j) -= mxB(i, j);
        }
    }
    
    return *this;
}

//
// Matrix binary operations
//

Matrix operator+(const Matrix &mxA, const Matrix &mxB)
{
    Matrix newMx(mxA.dimN(), mxA.dimM(), false);
    
    for (uint64_t i = 0; i < mxA.dimN(); ++i)
    {
        for (uint64_t j = 0; j < mxA.dimM(); ++j)
        {
            newMx(i, j) =
                    mxA(i, j) + mxB(i, j);
        }
    }
    
    return newMx;
}

Matrix operator-(const Matrix &mxA, const Matrix &mxB)
{
    Matrix newMx(mxA.dimN(), mxA.dimM(), false);
    
    for (uint64_t i = 0; i < mxA.dimN(); ++i)
    {
        for (uint64_t j = 0; j < mxA.dimM(); ++j)
        {
            newMx(i, j) =
                    mxA(i, j) - mxB(i, j);
        }
    }
    
    return newMx;
}

Matrix &operator*(double scalar, Matrix &mx)
{
    return mx * scalar;
}

Matrix &operator*(Matrix &mx, double scalar)
{
    for (uint64_t i = 0; i < mx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < mx.dimM(); ++j)
        {
            mx(i, j) *= scalar;
        }
    }
    
    return mx;
}

// Single-threaded

#ifndef multi

Matrix operator*(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimM() == mxB.dimN());
    
    Matrix newMx(mxA.dimN(), mxB.dimM(), false);
    uint64_t sharedDim = mxA.dimM();
    double temp;
    
    for (uint64_t i = 0; i < newMx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < newMx.dimM(); ++j)
        {
            temp = 0.0;
            
            for (uint64_t k = 0; k < sharedDim; ++k)
            {
                temp += mxA(i, k) * mxB(k, j);
            }
            
            newMx(i, j) = temp;
        }
    }
    
    
    return newMx;
}

Matrix matrix_mult_AT_B(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimN() == mxB.dimN());
    
    Matrix newMx(mxA.dimM(), mxB.dimM(), false);
    uint64_t sharedDim = mxA.dimN();
    double temp;
    
    for (uint64_t i = 0; i < newMx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < newMx.dimM(); ++j)
        {
            temp = 0.0;
            
            for (uint64_t k = 0; k < sharedDim; ++k)
            {
                temp += mxA(k, i) * mxB(k, j);
            }
            
            newMx(i, j) = temp;
        }
    }
    
    return newMx;
}

Matrix matrix_mult_A_BT(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimM() == mxB.dimM());
    
    Matrix newMx(mxA.dimN(), mxB.dimN(), false);
    uint64_t sharedDim = mxA.dimM();
    double temp;
    
    for (uint64_t i = 0; i < newMx.dimN(); ++i)
    {
        for (uint64_t j = 0; j < newMx.dimM(); ++j)
        {
            temp = 0.0;
            
            for (uint64_t k = 0; k < sharedDim; ++k)
            {
                temp += mxA(i, k) * mxB(j, k);
            }
            
            newMx(i, j) = temp;
        }
    }
    
    return newMx;
}

#endif

//
// Matrix x Vector operations
//

// Single-threaded

#ifndef multi

Vector operator*(const Matrix &mx, const Vector &vec)
{
    assert(mx.dimM() == vec.dim());
    
    // M * v
    Vector newVec(mx.dimN(), false);
    
    for (uint64_t i = 0; i < newVec.dim(); ++i)
    {
        double temp = mx(i, 0) * vec[0];
        for (uint64_t j = 1; j < mx.dimM(); ++j)
        {
            temp += mx(i, j) * vec[j];
        }
        newVec[i] = temp;
    }
    
    return newVec;
}

Vector operator^(const Matrix &mx_t, const Vector &vec)
{
    assert(mx_t.dimN() == vec.dim());
    
    // M^T * v
    Vector newVec(mx_t.dimM(), true);
    
    // maintain row-first iteration order for matrix
    for (uint64_t i = 0; i < mx_t.dimN(); ++i)
    {
        for (uint64_t j = 0; j < newVec.dim(); ++j)
        {
            newVec[j] += mx_t(i, j) * vec[i];
        }
    }
    
    return newVec;
}

Matrix vector_outer(const Vector &vecA, const Vector &vecB)
{
    //outer product, aka a * b^T, sizes don't matter!
    Matrix newMx(vecA.dim(), vecB.dim(), false);
    
    for (uint64_t i = 0; i < vecA.dim(); ++i)
    {
        for (uint64_t j = 0; j < vecB.dim(); ++j)
        {
            newMx(i, j) = vecA[i] * vecB[j];
        }
    }
    
    return newMx;
}

#endif

//
// A collection of all parallel versions of the operations
//
#ifdef multi

Matrix operator*(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimM() == mxB.dimN());
    
    Matrix newMx(mxA.dimN(), mxB.dimM(), false);
    uint64_t sharedDim = mxA.dimM();
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newMx.dimN() / total;
        uint64_t myend = (myid + 1) * newMx.dimN() / total;
        
        double temp;
        
        for (uint64_t i = mystart; i < myend; ++i)
        {
            for (uint64_t j = 0; j < newMx.dimM(); ++j)
            {
                temp = 0.0;
                
                for (uint64_t k = 0; k < sharedDim; ++k)
                {
                    temp += mxA(i, k) * mxB(k, j);
                }
                
                newMx(i, j) = temp;
            }
        }
    }
    
    return newMx;
}

Matrix matrix_mult_AT_B(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimN() == mxB.dimN());
    
    Matrix newMx(mxA.dimM(), mxB.dimM(), false);
    uint64_t sharedDim = mxA.dimN();
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newMx.dimN() / total;
        uint64_t myend = (myid + 1) * newMx.dimN() / total;
        
        double temp;
        
        for (uint64_t i = mystart; i < myend; ++i)
        {
            for (uint64_t j = 0; j < newMx.dimM(); ++j)
            {
                temp = 0.0;
                
                for (uint64_t k = 0; k < sharedDim; ++k)
                {
                    temp += mxA(k, i) * mxB(k, j);
                }
                
                newMx(i, j) = temp;
            }
        }
    }
    
    return newMx;
}

Matrix matrix_mult_A_BT(const Matrix &mxA, const Matrix &mxB)
{
    assert(mxA.dimM() == mxB.dimM());
    
    Matrix newMx(mxA.dimN(), mxB.dimN(), false);
    uint64_t sharedDim = mxA.dimM();
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newMx.dimN() / total;
        uint64_t myend = (myid + 1) * newMx.dimN() / total;
        
        double temp;
        
        for (uint64_t i = mystart; i < myend; ++i)
        {
            for (uint64_t j = 0; j < newMx.dimM(); ++j)
            {
                temp = 0.0;
                
                for (uint64_t k = 0; k < sharedDim; ++k)
                {
                    temp += mxA(i, k) * mxB(j, k);
                }
                
                newMx(i, j) = temp;
            }
        }
    }
    
    return newMx;
}

//
// Matrix x Vector operations
//

Vector operator*(const Matrix &mx, const Vector &vec)
{
    assert(mx.dimM() == vec.dim());
    
    // M * v
    Vector newVec(mx.dimN(), false);
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newVec.dim() / total;
        uint64_t myend = (myid + 1) * newVec.dim() / total;
        
        for (uint64_t i = mystart; i < myend; ++i)
        {
            double temp = mx(i, 0) * vec[0];
            for (uint64_t j = 1; j < mx.dimM(); ++j)
            {
                temp += mx(i, j) * vec[j];
            }
            newVec[i] = temp;
        }
    }
    
    return newVec;
}

Vector operator^(const Matrix &mx_t, const Vector &vec)
{
    assert(mx_t.dimN() == vec.dim());
    
    // M^T * v
    Vector newVec(mx_t.dimM(), true);
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newVec.dim() / total;
        uint64_t myend = (myid + 1) * newVec.dim() / total;
        
        double temp;
        
        // maintain row-first iteration order for matrix
        for (uint64_t j = mystart; j < myend; ++j)
        {
            temp = mx_t(0, j) * vec[j];
            for (uint64_t i = 0; i < mx_t.dimN(); ++i)
            {
                temp += mx_t(i, j) * vec[i];
            }
            newVec[j] = temp;
        }
    }
    
    return newVec;
}

Matrix vector_outer(const Vector &vecA, const Vector &vecB)
{
    //outer product, aka a * b^T, sizes don't matter!
    Matrix newMx(vecA.dim(), vecB.dim(), false);
    
  #pragma omp parallel
    {
        uint64_t myid = (uint64_t)omp_get_thread_num();
        uint64_t total = (uint64_t)omp_get_num_threads();
        
        uint64_t mystart = myid * newMx.dimN() / total;
        uint64_t myend = (myid + 1) * newMx.dimN() / total;
        
        for (uint64_t i = mystart; i < myend; ++i)
        {
            for (uint64_t j = 0; j < newMx.dimM(); ++j)
            {
                newMx(i, j) = vecA[i] * vecB[j];
            }
        }
    }
    
    return newMx;
}

#endif

} // namespace Algebra
