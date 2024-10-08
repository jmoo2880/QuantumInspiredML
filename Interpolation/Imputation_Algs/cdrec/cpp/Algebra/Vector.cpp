//
// Created by Zakhar on 05.03.2017.
//

#include <cmath>
#include <cassert>

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Vector.h"
#include "Matrix.h"

namespace Algebra
{

//
// Vector data
//

double &Vector::operator[](uint64_t i)
{
    return _data[i];
}

double &Vector::operator[](uint64_t i) const
{
    return _data[i];
}

uint64_t Vector::dim() const
{ return _dim; }

Vector Vector::copy() const
{
    return Vector(_dim, _data, true);
}

std::string Vector::toString() const
{
    std::stringstream strbuffer;
    
    strbuffer << std::fixed << std::setprecision(2) << "(";
    for (uint64_t i = 0; i < _dim - 1; i++)
    {
        double elem = operator[](i);
        strbuffer << (elem >= 0 ? " " : "") << elem << "\t";
    }
    strbuffer << operator[](_dim - 1) << ")^T" << std::endl;
    
    return strbuffer.str();
}

void Vector::expandStorage()
{
    uint64_t newDim = _dim + 1;
    
    if (newDim > _dimReal)
    {
        // reallocate memory
        _dimReal = _dim < 4 ? 4 : static_cast<size_t>(std::round((double)_dimReal * 1.4));
        _data = (double *)std::realloc(_data, _dimReal * sizeof(*_data));
    }
    
    _dim = newDim;
}

//
// Vector constructors
//

Vector::Vector()
        : _dimReal(0),
          _isReference(false),
          _data(nullptr),
          _dim(0)
{ }

Vector::Vector(uint64_t n, bool init)
        : _dimReal(n),
          _dim(n)
{
    _data = (double *)std::malloc(_dim * sizeof(*_data));
    
    if (init)
    {
        std::fill(_data, _data + _dim, 0);
    }
}

Vector::Vector(uint64_t n, double *arr, bool copy)
        : _dimReal(n),
          _isReference(!copy),
          _dim(n)
{
    if (copy)
    {
        _data = (double *)std::malloc(_dim * sizeof(*_data));
        std::copy(arr, arr + _dim, _data);
    }
    else
    {
        _data = arr;
    }
}

Vector::Vector(const std::vector<double> &data)
        : _dimReal(data.size()),
          _dim(data.size())
{
    _data = (double *)std::malloc(_dim * sizeof(*_data));
    
    for (uint64_t i = 0; i < _dim; i++)
    {
        _data[i] = data[i];
    }
}

Vector::Vector(Vector &&other) noexcept
        : _dimReal(other._dimReal),
          _isReference(other._isReference),
          _data(other._data),
          _dim(other._dim)
{
    other._dimReal = 0;
    other._isReference = false;
    other._data = nullptr;
    other._dim = 0;
}

Vector &Vector::operator=(Vector &&other) noexcept
{
    if (this != &other)
    {
        if (_data != nullptr && !_isReference)
        { free(_data); }
        
        _dimReal = other._dimReal;
        _isReference = other._isReference;
        _data = other._data;
        _dim = other._dim;
        
        other._dimReal = 0;
        other._isReference = false;
        other._data = nullptr;
        other._dim = 0;
    }
    
    return *this;
}

Vector::~Vector()
{
    if (_data != nullptr && !_isReference)
    {
        std::free(_data);
    }
}

//
// Pre-defined vectors
//

Vector Vector::canonical(uint64_t dim, uint64_t pos)
{
    Vector vec(dim, true);
    
    vec[pos] = 1.0;
    
    return vec;
}

Vector Vector::empty()
{
    return Vector();
}

Vector *Vector::emptyPtr()
{
    return new Vector();
}

//
// Vector operations
//

double Vector::norm2()
{
    return sqrt(vector_dot(*this, *this));
}

Vector &Vector::normalize()
{
    return (*this) / norm2();// todo: wtf is that????
}

Vector &Vector::fill(double val)
{
    std::fill_n(_data, _dim, val);
    
    return (*this);
}

Vector &Vector::append(double value)
{
    if (_isReference)
    {
        throw ReferenceAppendedException("Appending to reference-vector is not allowed, create a copy() first");
    }
    
    expandStorage();
    
    operator[](_dim - 1) = value;
    
    return *this;
}

Vector Vector::subVector(uint64_t size)
{
    assert(size <= _dim);
    
    return Vector(size, _data, true);
}


Vector Vector::subVector(uint64_t start, uint64_t size)
{
    assert(start + size <= _dim);
    
    return Vector(size, _data + start, true);
}

Matrix Vector::diag()
{
    Matrix newMat(_dim, _dim, true);
    
    for (uint64_t i = 0; i < _dim; ++i)
    {
        newMat(i, i) = operator[](i);
    }
    
    return newMat;
}

//
// Vector operators
//

Vector operator+(const Vector &vecA, const Vector &vecB)
{
    Vector newVec(vecA.dim(), false);
    
    for (uint64_t i = 0; i < vecA.dim(); ++i)
    {
        newVec[i] = vecA[i] + vecB[i];
    }
    
    return newVec;
}

Vector operator-(const Vector &vecA, const Vector &vecB)
{
    Vector newVec(vecA.dim(), false);
    
    for (uint64_t i = 0; i < vecA.dim(); ++i)
    {
        newVec[i] = vecA[i] - vecB[i];
    }
    
    return newVec;
}

Vector &operator*(Vector &vec, double scalar)
{
    for (uint64_t i = 0; i < vec.dim(); ++i)
    {
        vec[i] *= scalar;
    }
    return vec;
}

Vector &operator/(Vector &vec, double scalar)
{
    for (uint64_t i = 0; i < vec.dim(); ++i)
    {
        vec[i] /= scalar;
    }
    return vec;
}

Vector &operator*(double scalar, Vector &vec)
{
    return vec * scalar;
}

double vector_dot(const Vector &vec1, const Vector &vec2)
{
    assert(vec1.dim() == vec2.dim());
    double dot = 0.0;
    
    for (uint64_t i = 0; i < vec1.dim(); ++i)
    {
        dot += vec1[i] * vec2[i];
    }
    
    return dot;
}

} // namespace Algebra
