#ifndef CUDAMLP_CPUMATRIX_CUH
#define CUDAMLP_CPUMATRIX_CUH

#include <iostream>
#include <iomanip>
#include <exception>
#include <random>
#include <array>

class CPUMatrix {
    friend std::ostream &operator<<(std::ostream &out, const CPUMatrix &m) {
        out << "CPUMatrix of " << m.row << " x " << m.col;
        return out;
    }

    static float **malloc_matrix(const unsigned &r, const unsigned &c) {
        auto m = new float *[r];
        for (auto i = 0; i < r; i++)
            m[i] = new float[c];
        return m;
    }

public:
    CPUMatrix() = default;

    CPUMatrix(const unsigned &r, const unsigned &c, const float &val = 0.0) : CPUMatrix() {
        row = r;
        col = c;
        data = malloc_matrix(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] = val;
    }

    CPUMatrix(const CPUMatrix &m) : CPUMatrix(m.row, m.col) {
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] = m.data[i][j];
    }

    CPUMatrix(const std::initializer_list<std::initializer_list<float>> &arr) : CPUMatrix() {
        if (arr.size() == 0 || arr.begin()->size() == 0) return;

        auto arr_n_rows = arr.size(), arr_n_cols = arr.begin()->size();
        for (auto it = arr.begin(); it != arr.end(); it++)
            if (it->size() != arr_n_cols)
                throw std::runtime_error("Not a matrix.");

        row = arr_n_rows;
        col = arr_n_cols;
        data = malloc_matrix(row, col);
        unsigned i = 0, j;
        for (auto row_it = arr.begin(); row_it < arr.end(); i++, row_it++) {
            j = 0;
            for (auto col_it = row_it->begin(); col_it < row_it->end(); j++, col_it++)
                data[i][j] = *col_it;
        }
    }

    ~CPUMatrix() {
        clear();
    }

    // Operators
    CPUMatrix &operator=(const CPUMatrix &m) {
        if (&m == this) return *this;
        if (row && col)
            clear();

        row = m.row;
        col = m.col;
        data = malloc_matrix(row, col);

        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] = m(i, j);
        return *this;
    }

    CPUMatrix &operator=(const std::initializer_list<std::initializer_list<float>> &arr) {
        CPUMatrix _m(arr);
        return (*this = _m);
    }

    CPUMatrix operator+(const CPUMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] + m(i, j);
        return n;
    }

    CPUMatrix operator-(const CPUMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] - m(i, j);
        return n;
    }

    CPUMatrix operator*(const CPUMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] * m(i, j);
        return n;
    }

    CPUMatrix operator/(const CPUMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] / m(i, j);
        return n;
    }

    CPUMatrix operator+(const float &scalar) const {
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] + scalar;
        return n;
    }

    CPUMatrix operator-(const float &scalar) const {
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] - scalar;
        return n;
    }

    CPUMatrix operator*(const float &scalar) const {
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] * scalar;
        return n;
    }

    CPUMatrix operator/(const float &scalar) const {
        CPUMatrix n(row, col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(i, j) = data[i][j] / scalar;
        return n;
    }

    CPUMatrix operator%(const CPUMatrix &m) const {
        if (col != m.row)
            throw std::runtime_error("Shape is not compatible. Column should equal to row.");
        CPUMatrix n(row, m.col);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < m.col; j++)
                for (auto k = 0; k < col; k++)
                    n(i, j) += data[i][k] * m(k, j);
        return n;
    }

    CPUMatrix &operator+=(const CPUMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] += m(i, j);
        return (*this);
    }

    CPUMatrix &operator-=(const CPUMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] -= m(i, j);
        return (*this);
    }

    CPUMatrix &operator*=(const CPUMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] *= m(i, j);
        return (*this);
    }

    CPUMatrix &operator/=(const CPUMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] /= m(i, j);
        return (*this);
    }

    CPUMatrix &operator+=(const float &scalar) {
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] += scalar;
        return (*this);
    }

    CPUMatrix &operator-=(const float &scalar) {
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] -= scalar;
        return (*this);
    }

    CPUMatrix &operator*=(const float &scalar) {
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] *= scalar;
        return (*this);
    }

    CPUMatrix &operator/=(const float &scalar) {
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                data[i][j] /= scalar;
        return (*this);
    }

    CPUMatrix &operator%=(const CPUMatrix &m) {
        CPUMatrix n = (*this) % m;
        (*this) = n;
        return (*this);
    }

    float &operator()(const unsigned &r, const unsigned &c) {
        return this->data[r][c];
    }

    const float &operator()(const unsigned &r, const unsigned &c) const {
        return this->data[r][c];
    }

    CPUMatrix transpose() {
        CPUMatrix n(col, row);
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                n(j, i) = data[i][j];
        return n;
    }

    unsigned n_rows() const {
        return row;
    }

    unsigned n_cols() const {
        return col;
    }

    std::pair<unsigned, unsigned> shape() const {
        return std::make_pair(row, col);
    }

    float sum() const {
        float s = 0;
        for (auto i = 0; i < row; i++)
            for (auto j = 0; j < col; j++)
                s += data[i][j];
        return s;
    }

    void clear() {
        if (!row || !col) return;
        for (auto i = 0; i < row; i++)
            delete[] data[i];
        delete[] data;

        row = col = 0;
        data = nullptr;
    }

    void print_all_numbers(bool endl = true) const {
        for (auto i = 0; i < row; ++i) {
            for (auto j = 0; j < col; ++j)
                std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                          << data[i][j];
            std::cout << std::endl;
        }
        if (endl) std::cout << std::endl;
    }

private:
    unsigned row{0};
    unsigned col{0};
    float **data{nullptr};
};

// sigmoid functions
CPUMatrix sigmoid(const CPUMatrix &m) {
    CPUMatrix _m(m.n_rows(), m.n_cols());
    for (auto i = 0; i < _m.n_rows(); i++)
        for (auto j = 0; j < _m.n_cols(); j++)
            _m(i, j) = 1 / (1 + exp(-m(i, j)));
    return _m;
}

CPUMatrix d_sigmoid(const CPUMatrix &m) {
    return m * (m * (-1) + 1);
}

#endif //CUDAMLP_CPUMATRIX_CUH
