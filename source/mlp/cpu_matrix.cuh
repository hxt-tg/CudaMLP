#ifndef CUDAMLP_CPU_MATRIX_CUH
#define CUDAMLP_CPU_MATRIX_CUH

#include <iostream>
#include <iomanip>
#include <exception>
#include <random>
#include <array>
#include "matrix.cuh"

class CPUMatrix : public BaseMatrix<CPUMatrix> {
    friend std::ostream &operator<<(std::ostream &out, const CPUMatrix &m) {
        out << "CPUMatrix of " << m._row << " x " << m._col;
        return out;
    }

    friend CPUMatrix sigmoid(const CPUMatrix &m);

public:
    CPUMatrix() = default;

    CPUMatrix(const unsigned &r, const unsigned &c, const float &val = 0.0) : CPUMatrix() {
        if (!r || !c) return;
        _row = r;
        _col = c;
        _size = _row * _col * sizeof(float);
        _data = new float[_row * _col];
        for (auto i = 0; i < _row * _col; i++)
            _data[i] = val;
    }

    CPUMatrix(const CPUMatrix &m) : CPUMatrix(m._row, m._col) {
        memcpy(_data, m._data, _size);
    }

    CPUMatrix(const std::initializer_list<std::initializer_list<float>> &arr) : CPUMatrix() {
        if (arr.size() == 0 || arr.begin()->size() == 0) return;

        auto arr_n_rows = arr.size(), arr_n_cols = arr.begin()->size();
        for (auto it = arr.begin(); it != arr.end(); it++)
            if (it->size() != arr_n_cols)
                throw std::runtime_error("Not a matrix.");

        _row = arr_n_rows;
        _col = arr_n_cols;
        _size = _row * _col * sizeof(float);
        _data = new float[_row * _col];
        unsigned i = 0;
        for (auto row_it = arr.begin(); row_it < arr.end(); i++, row_it++)
            for (auto col_it = row_it->begin(); col_it < row_it->end(); i++, col_it++)
                _data[i] = *col_it;
    }

    ~CPUMatrix() {
        clear();
    }

    void clear() {
        if (!_size) return;
        delete[] _data;
        _row = _col = _size = 0;
        _data = nullptr;
    }

    // Operators
    CPUMatrix &operator=(const CPUMatrix &m) {
        if (&m == this) return *this;
        if (_row && _col)
            clear();

        _row = m._row;
        _col = m._col;
        _size = _row * _col * sizeof(float);
        _data = new float[_row * _col];
        memcpy(_data, m._data, _size);
        return *this;
    }

    CPUMatrix &operator=(const std::initializer_list<std::initializer_list<float>> &arr) {
        CPUMatrix _m(arr);
        return (*this = _m);
    }

    CPUMatrix operator+(const CPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] + m._data[i];
        return _m;
    }

    CPUMatrix operator-(const CPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix n(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            n._data[i] = _data[i] - m._data[i];
        return n;
    }

    CPUMatrix operator*(const CPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] * m._data[i];
        return _m;
    }

    CPUMatrix operator/(const CPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] / m._data[i];
        return _m;
    }

    CPUMatrix operator+(const float &scalar) const override {
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] + scalar;
        return _m;
    }

    CPUMatrix operator-(const float &scalar) const override {
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] - scalar;
        return _m;
    }

    CPUMatrix operator*(const float &scalar) const override {
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] * scalar;
        return _m;
    }

    CPUMatrix operator/(const float &scalar) const override {
        CPUMatrix _m(_row, _col);
        for (auto i = 0; i < _row * _col; i++)
            _m._data[i] = _data[i] / scalar;
        return _m;
    }

    CPUMatrix operator%(const CPUMatrix &m) const override {
        if (_col != m._row)
            throw std::runtime_error("Shape is not compatible. Column should equal to _row.");
        CPUMatrix _m(_row, m._col);
        for (auto i = 0; i < _row; i++)
            for (auto j = 0; j < m._col; j++)
                for (auto k = 0; k < _col; k++)
                    _m._data[i * m._col + j] += _data[i * _col + k] * m._data[k * m._col + j];
        return _m;
    }

    CPUMatrix &operator+=(const CPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < _row * _col; i++)
            _data[i] += m._data[i];
        return (*this);
    }

    CPUMatrix &operator-=(const CPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < _row * _col; i++)
            _data[i] -= m._data[i];
        return (*this);
    }

    CPUMatrix &operator*=(const CPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < _row * _col; i++)
            _data[i] *= m._data[i];
        return (*this);
    }

    CPUMatrix &operator/=(const CPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < _row * _col; i++)
            _data[i] /= m._data[i];
        return (*this);
    }

    CPUMatrix &operator+=(const float &scalar) override {
        for (auto i = 0; i < _row * _col; i++)
            _data[i] += scalar;
        return (*this);
    }

    CPUMatrix &operator-=(const float &scalar) override {
        for (auto i = 0; i < _row * _col; i++)
            _data[i] -= scalar;
        return (*this);
    }

    CPUMatrix &operator*=(const float &scalar) override {
        for (auto i = 0; i < _row * _col; i++)
            _data[i] *= scalar;
        return (*this);
    }

    CPUMatrix &operator/=(const float &scalar) override {
        for (auto i = 0; i < _row * _col; i++)
            _data[i] /= scalar;
        return (*this);
    }

    CPUMatrix &operator%=(const CPUMatrix &m) override {
        CPUMatrix _m = (*this) % m;
        (*this) = _m;
        return (*this);
    }

    float &operator()(const unsigned &r, const unsigned &c) {
        return this->_data[r * _col + c];
    }

    const float &operator()(const unsigned &r, const unsigned &c) const {
        return this->_data[r * _col + c];
    }

    CPUMatrix transpose() const override {
        CPUMatrix _m(_col, _row);
        for (auto i = 0; i < _row; i++)
            for (auto j = 0; j < _col; j++)
                _m._data[j * _row + i] = _data[i * _col + j];
        return _m;
    }

    float sum() const override {
        float s = 0;
        for (auto i = 0; i < _row * _col; i++)
            s += _data[i];
        return s;
    }

    unsigned row() const {
        return _row;
    }

    unsigned col() const {
        return _col;
    }

    void print_all_numbers(bool endl = true) const {
        for (auto i = 0; i < _row; ++i) {
            for (auto j = 0; j < _col; ++j)
                std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                          << _data[i * _col + j];
            std::cout << std::endl;
        }
        if (endl) std::cout << std::endl;
    }

private:
    unsigned _row{0};
    unsigned _col{0};
    size_t _size{0};
    float *_data{nullptr};
    friend class GPUMatrix;
};

// sigmoid functions
CPUMatrix sigmoid(const CPUMatrix &m) {
    CPUMatrix _m(m._row, m._col);
    for (auto i = 0; i < _m._row * _m._col; i++)
        _m._data[i] = 1 / (1 + exp(-m._data[i]));
    return _m;
}

CPUMatrix d_sigmoid(const CPUMatrix &m) {
    return m * (m * (-1) + 1);
}

#endif //CUDAMLP_CPU_MATRIX_CUH
