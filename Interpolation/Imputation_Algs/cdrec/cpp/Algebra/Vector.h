//
// Created by Zakhar on 07.03.2017.
//

#pragma once

#include <vector>
#include <string>

namespace Algebra
{

class Matrix;

class Vector
{
    friend class Matrix;
    //
    // Data
    //
  private:
    uint64_t _dimReal;
    bool _isReference = false;
  
  protected:
    double *_data;
    uint64_t _dim;
  
  public:
    uint64_t dim() const;
    
    double &operator[](uint64_t i);
    
    double &operator[](uint64_t i) const;
    
    /// Creates a copy of the vector with all its data
    Vector copy() const;
    
    /// Generates a string representation of the vector
    std::string toString() const;
  
  private:
    /// Increases vector's capacity by 1
    void expandStorage();
    
    //
    // Constructors and destructors
    //
  private:
    explicit Vector();
  
  public:
    /// Default constructor - creates a vector of n 0.0 values
    /// @param [in] n vector size
    /// @param [in] init if constructor is to zerofill the vector
    Vector(uint64_t n, bool init);
    
    /// Construct a vector from existing data
    /// @param [in] n vector size
    /// @param [in] arr data to construct a vector from
    /// @param [in] copy to copy the data or to keep the provided pointer
    Vector(uint64_t n, double *arr, bool copy);
    
    /// Construct a vector from std::vector
    /// @param [in] data std::vector of data to construct a vector from
    explicit Vector(const std::vector<double> &data);
    
    /// Not copy-constructible
    Vector(const Vector &other) = delete;
    
    Vector(Vector &other) = delete;
    
    Vector &operator=(const Vector &other) = delete;
    
    Vector &operator=(Vector &other) = delete;
    
    /// Move constructor
    /// @param [in] other vector to move from
    Vector(Vector &&other) noexcept;
    
    /// Move assignment
    /// @param [in] other vector to move from
    Vector &operator=(Vector &&other) noexcept;
    
    /// Destroys the vector.
    /// Frees the memory of data container if it wasn't a reference
    ~Vector();
    
    //
    // Pre-defined vectors
    //
  public:
    /// Creates an R^n canonical unit vector
    /// @param [in] dim vector size
    /// @param [in] pos a position to contain a unit element
    static Vector canonical(uint64_t dim, uint64_t pos);
    
    static Vector empty();
    
    static Vector *emptyPtr();
    
    //
    // Vector operations
    //
  public:
    /// Returns an euclidean norm of the current Vector
    double norm2();
    
    /// Normalizes the current vector by it's euclidean norm
    Vector &normalize();
    
    Vector &fill(double val);
    
    /// Increases the storage capacity of the Vector by 1 and places the value at the added position
    /// @param [in] value element inserted at a new position
    Vector &append(double value);
    
    /// Creates a new vector by taking first `size` elements of the current one
    /// @param [in] size amount of elements to take for a new vector
    Vector subVector(uint64_t size);
    
    /// Creates a new vector by taking `size` elements of the current one starting with `start`
    /// @param [in] start index of the first element to take for a new vector
    /// @param [in] size amount of elements to take for a new vector
    Vector subVector(uint64_t start, uint64_t size);
    
    /// Constructs a square matrix with the currect vector on the diagonal
    Matrix diag();
}; // end class

//
// Operators
//

Vector operator+(const Vector &vecA, const Vector &vecB);

Vector operator-(const Vector &vecA, const Vector &vecB);

Vector &operator*(Vector &vec, double scalar);

Vector &operator*(double scalar, Vector &vec);

Vector &operator/(Vector &vec, double scalar);

double vector_dot(const Vector &vec1, const Vector &vec2);

//
// Error handling
//

struct ReferenceAppendedException : public std::exception
{
  private:
    const char *message;
  
  public:
    explicit ReferenceAppendedException(const char *what_arg)
            : message(what_arg)
    { }
    
    const char *what() const noexcept override
    {
        return message;
    }
};

} // namespace Algebra
