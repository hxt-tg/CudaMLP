#ifndef CUDAMLP_MATRIX_CUH
#define CUDAMLP_MATRIX_CUH

template<typename MatrixClass>
class BaseMatrix {
public:
//    virtual MatrixClass &operator=(const MatrixClass &m) = 0;

//    produce new instance
    virtual MatrixClass operator+(const MatrixClass &m) const = 0;

    virtual MatrixClass operator-(const MatrixClass &m) const = 0;

    virtual MatrixClass operator*(const MatrixClass &m) const = 0;

    virtual MatrixClass operator/(const MatrixClass &m) const = 0;

    virtual MatrixClass operator+(const float &scalar) const = 0;

    virtual MatrixClass operator-(const float &scalar) const = 0;

    virtual MatrixClass operator*(const float &scalar) const = 0;

    virtual MatrixClass operator/(const float &scalar) const = 0;

    virtual MatrixClass operator%(const MatrixClass &m) const = 0;

//    In-place calculation
    virtual MatrixClass &operator+=(const MatrixClass &m) = 0;

    virtual MatrixClass &operator-=(const MatrixClass &m) = 0;

    virtual MatrixClass &operator*=(const MatrixClass &m) = 0;

    virtual MatrixClass &operator/=(const MatrixClass &m) = 0;

    virtual MatrixClass &operator+=(const float &scalar) = 0;

    virtual MatrixClass &operator-=(const float &scalar) = 0;

    virtual MatrixClass &operator*=(const float &scalar) = 0;

    virtual MatrixClass &operator/=(const float &scalar) = 0;

    virtual MatrixClass &operator%=(const MatrixClass &m) = 0;

    virtual MatrixClass transpose() const = 0;

    virtual float sum() const = 0;

};

class CPUMatrix;
class GPUMatrix;

#endif //CUDAMLP_MATRIX_CUH
