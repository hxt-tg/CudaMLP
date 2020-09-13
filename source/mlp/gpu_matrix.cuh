#ifndef CUDAMLP_GPU_MATRIX_CUH
#define CUDAMLP_GPU_MATRIX_CUH

#include <iostream>
#include <iomanip>
#include <exception>
#include <random>
#include <array>
#include <cuda.h>
#include <curand.h>
#include "matrix.cuh"
#include "cpu_matrix.cuh"

#define BLOCK_SIZE  32

/* ======== CUDA check ======== */
#define CudaCheck(status) CudaErrorHandler((status), __FILE__, __LINE__)

inline void CudaErrorHandler(cudaError_t status, const char *file, int line, bool interrupt = true) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "CUDA error %d \"%s\" at %s %d\n", status, cudaGetErrorString(status), file, line);
    if (interrupt) exit(status);
}

#define CurandCheck(status) CurandErrorHandler((status), __FILE__, __LINE__)

inline void CurandErrorHandler(curandStatus_t status, const char *file, int line, bool interrupt = true) {
    if (status == CURAND_STATUS_SUCCESS) return;
    fprintf(stderr, "CURAND error %d at %s %d\n", status, file, line);
    if (interrupt) exit(status);
}

/* ======== Kernel functions ======== */
typedef float (*kernel_ptr)(float, float);

__device__ float kernel_add(float x, float y) { return x + y; }

__device__ float kernel_sub(float x, float y) { return x - y; }

__device__ float kernel_mul(float x, float y) { return x * y; }

__device__ float kernel_div(float x, float y) { return x / y; }

typedef enum {
    KERNEL_ADD,
    KERNEL_SUB,
    KERNEL_MUL,
    KERNEL_DIV,
} KernelFuncType;

__device__ kernel_ptr device_kernel_add = kernel_add;
__device__ kernel_ptr device_kernel_sub = kernel_sub;
__device__ kernel_ptr device_kernel_mul = kernel_mul;
__device__ kernel_ptr device_kernel_div = kernel_div;


/* ======== Element-wise kernels ======== */
__global__ void element_wise_kernel(float *result, const float *a, const float *b, unsigned row, unsigned col,
                                    kernel_ptr calc) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= row || j >= col) return;
    result[i * col + j] = calc(a[i * col + j], b[i * col + j]);
}

void element_wise_interface(float *result, const float *a, const float *b, unsigned row, unsigned col,
                            KernelFuncType op) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    kernel_ptr op_func;
    CudaCheck(cudaMalloc(&op_func, sizeof(kernel_ptr)));
    switch (op) {
        case KERNEL_ADD:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_add, sizeof(kernel_ptr)));
            break;
        case KERNEL_SUB:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_sub, sizeof(kernel_ptr)));
            break;
        case KERNEL_MUL:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_mul, sizeof(kernel_ptr)));
            break;
        case KERNEL_DIV:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_div, sizeof(kernel_ptr)));
            break;
        default:
            throw std::runtime_error("Unsupported kernel operation.");
    }
    element_wise_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, row, col, *op_func);
    CudaCheck(cudaDeviceSynchronize());
}


/* ======== Scalar kernels ======== */
__global__ void scalar_kernel(float *result, const float *a, const float b, unsigned row, unsigned col,
                              kernel_ptr calc) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= row || j >= col) return;
    result[i * col + j] = calc(a[i * col + j], b);
}

void scalar_interface(float *result, const float *a, const float b, unsigned row, unsigned col,
                      KernelFuncType op) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    kernel_ptr op_func;
    CudaCheck(cudaMalloc(&op_func, sizeof(kernel_ptr)));
    switch (op) {
        case KERNEL_ADD:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_add, sizeof(kernel_ptr)));
            break;
        case KERNEL_SUB:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_sub, sizeof(kernel_ptr)));
            break;
        case KERNEL_MUL:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_mul, sizeof(kernel_ptr)));
            break;
        case KERNEL_DIV:
            CudaCheck(cudaMemcpyFromSymbol(&op_func, device_kernel_div, sizeof(kernel_ptr)));
            break;
        default:
            throw std::runtime_error("Unsupported kernel operation.");
    }
    scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, row, col, *op_func);
    CudaCheck(cudaDeviceSynchronize());
}


/* ======== Matrix multiplication kernels ======== */
/*
 * Shape of a: _row x mid
 * Shape of b: mid x _col
 */
__global__ void matmul_kernel(float *result, const float *a, const float *b,
                              unsigned row, unsigned mid, unsigned col) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;
    if (i >= row || j >= col) return;

    for (auto k = 0; k < mid; k++)
        sum += a[i * mid + k] * b[k * col + j];
    result[i * col + j] = sum;
}

void matmul_interface(float *result, const float *a, const float *b, unsigned row, unsigned mid, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, row, mid, col);
    CudaCheck(cudaThreadSynchronize());
}


/* ======== Matrix transpose kernels ======== */
__global__ void transpose_kernel(float *result, const float *a, unsigned row, unsigned col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= row || j >= col) return;
    result[i * col + j] = a[j * row + i];
}

void transpose_interface(float *result, const float *a, unsigned row, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, row, col);
    CudaCheck(cudaDeviceSynchronize());
}


/* ======== Matrix initialize kernels ======== */
__global__ void initialize_kernel(float *result, const float val, unsigned row, unsigned col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= row || j >= col) return;
    result[i * col + j] = val;
}

void initialize_interface(float *result, const float val, unsigned row, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initialize_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, val, row, col);
    CudaCheck(cudaDeviceSynchronize());
}

class GPUMatrix : public BaseMatrix<GPUMatrix> {
    friend std::ostream &operator<<(std::ostream &out, const GPUMatrix &m) {
        out << "GPUMatrix of " << m._row << " x " << m._col;
        return out;
    }

    friend GPUMatrix sigmoid(const GPUMatrix &m);

public:
    GPUMatrix() = default;

    GPUMatrix(const unsigned &r, const unsigned &c, const float &val = 0.0) : GPUMatrix() {
        if (!r || !c) return;
        _row = r;
        _col = c;
        _size = _row * _col * sizeof(float);
        CudaCheck(cudaMalloc(&_data, _size));
        initialize_interface(_data, val, _row, _col);
    }

    GPUMatrix(const GPUMatrix &m) : GPUMatrix(m._row, m._col) {
        CudaCheck(cudaMemcpy(_data, m._data, _size, cudaMemcpyDeviceToDevice));
    }

    explicit GPUMatrix(const CPUMatrix &m) : GPUMatrix() {
        if (!m._row || !m._col) return;
        _row = m._row;
        _col = m._col;
        _size = _row * _col * sizeof(float);
        CudaCheck(cudaMalloc(&_data, _size));
        CudaCheck(cudaMemcpy(_data, m._data, _size, cudaMemcpyHostToDevice));
    }

    ~GPUMatrix() {
        clear();
    }

    void clear() {
        if (!_size) return;
        CudaCheck(cudaFree(_data));
        _row = _col = _size = 0;
        _data = nullptr;
    }

    // Operators
    GPUMatrix &operator=(const GPUMatrix &m) {
        if (&m == this) return *this;
        if (_row && _col)
            clear();

        _row = m._row;
        _col = m._col;
        _size = m._size;
        CudaCheck(cudaMalloc(&_data, _size));
        CudaCheck(cudaMemcpy(_data, m._data, _size, cudaMemcpyDeviceToDevice));
        return *this;
    }

//    GPUMatrix &operator=(const std::initializer_list<std::initializer_list<float>> &arr) {
//        GPUMatrix _m(arr);
//        return (*this = _m);
//    }

    GPUMatrix operator+(const GPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        GPUMatrix _m(_row, _col);
        element_wise_interface(_m._data, _data, m._data, _row, _col, KERNEL_ADD);
        return _m;
    }

    GPUMatrix operator-(const GPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        GPUMatrix _m(_row, _col);
        element_wise_interface(_m._data, _data, m._data, _row, _col, KERNEL_SUB);
        return _m;
    }

    GPUMatrix operator*(const GPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        GPUMatrix _m(_row, _col);
        element_wise_interface(_m._data, _data, m._data, _row, _col, KERNEL_MUL);
        return _m;
    }

    GPUMatrix operator/(const GPUMatrix &m) const override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        GPUMatrix _m(_row, _col);
        element_wise_interface(_m._data, _data, m._data, _row, _col, KERNEL_DIV);
        return _m;
    }

    GPUMatrix operator+(const float &scalar) const override {
        GPUMatrix _m(_row, _col);
        scalar_interface(_m._data, _data, scalar, _row, _col, KERNEL_ADD);
        return _m;
    }

    GPUMatrix operator-(const float &scalar) const override {
        GPUMatrix _m(_row, _col);
        scalar_interface(_m._data, _data, scalar, _row, _col, KERNEL_SUB);
        return _m;
    }

    GPUMatrix operator*(const float &scalar) const override {
        GPUMatrix _m(_row, _col);
        scalar_interface(_m._data, _data, scalar, _row, _col, KERNEL_MUL);
        return _m;
    }

    GPUMatrix operator/(const float &scalar) const override {
        GPUMatrix _m(_row, _col);
        scalar_interface(_m._data, _data, scalar, _row, _col, KERNEL_DIV);
        return _m;
    }

    GPUMatrix operator%(const GPUMatrix &m) const override {
        if (_col != m._row)
            throw std::runtime_error("Shape is not compatible. Column should equal to _row.");
        GPUMatrix _m(_row, m._col);
        matmul_interface(_m._data, _data, m._data, _row, _col, m._col);
        return _m;
    }

    GPUMatrix &operator+=(const GPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(_data, _data, m._data, _row, _col, KERNEL_ADD);
        return (*this);
    }

    GPUMatrix &operator-=(const GPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(_data, _data, m._data, _row, _col, KERNEL_SUB);
        return (*this);
    }

    GPUMatrix &operator*=(const GPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(_data, _data, m._data, _row, _col, KERNEL_MUL);
        return (*this);
    }

    GPUMatrix &operator/=(const GPUMatrix &m) override {
        if (_col != m._col || _row != m._row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(_data, _data, m._data, _row, _col, KERNEL_DIV);
        return (*this);
    }

    GPUMatrix &operator+=(const float &scalar) override {
        scalar_interface(_data, _data, scalar, _row, _col, KERNEL_ADD);
        return (*this);
    }

    GPUMatrix &operator-=(const float &scalar) override {
        scalar_interface(_data, _data, scalar, _row, _col, KERNEL_SUB);
        return (*this);
    }

    GPUMatrix &operator*=(const float &scalar) override {
        scalar_interface(_data, _data, scalar, _row, _col, KERNEL_MUL);
        return (*this);
    }

    GPUMatrix &operator/=(const float &scalar) override {
        scalar_interface(_data, _data, scalar, _row, _col, KERNEL_DIV);
        return (*this);
    }

    GPUMatrix &operator%=(const GPUMatrix &m) override {
        GPUMatrix _m = (*this) % m;
        (*this) = _m;
        return (*this);
    }

    GPUMatrix transpose() const override {
        unsigned new_row = _col, new_col = _row;
        GPUMatrix _m(new_row, new_col);
        transpose_interface(_m._data, _data, new_row, new_col);
        return _m;
    }

    float sum() const override {
        auto d = new float[_size];
        CudaCheck(cudaMemcpy(d, _data, _size, cudaMemcpyDeviceToHost));

        float s = 0;
        for (auto i = 0; i < _row * _col; i++)
            s += d[i];
        delete[] d;
        return s;
    }

    unsigned row() const {
        return _row;
    }

    unsigned col() const {
        return _col;
    }

    void print_all_numbers(bool endl = true) const {
        auto d = new float[_size];
        CudaCheck(cudaMemcpy(d, _data, _size, cudaMemcpyDeviceToHost));

        for (auto i = 0; i < _row; ++i) {
            for (auto j = 0; j < _col; ++j)
                std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                          << d[i * _col + j];
            std::cout << std::endl;
        }
        if (endl) std::cout << std::endl;
        delete[] d;
    }

private:
    unsigned _row{0};
    unsigned _col{0};
    size_t _size{0};
    float *_data{nullptr};
};


/* ======== Sigmoid kernels ======== */
__global__ void sigmoid_kernel(float *result, const float *a, unsigned row, unsigned col) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= row || j >= col) return;
    result[i * col + j] = 1 / (1 + exp(-a[i * col + j]));
}

void sigmoid_interface(float *result, const float *a, unsigned row, unsigned col) {
    size_t size = row * col * sizeof(float);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float *d_a, *d_result;
    CudaCheck(cudaMalloc(&d_a, size));
    CudaCheck(cudaMalloc(&d_result, size));
    CudaCheck(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_a, row, col);
    CudaCheck(cudaDeviceSynchronize());

    CudaCheck(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));
    CudaCheck(cudaFree(d_a));
    CudaCheck(cudaFree(d_result));
}

GPUMatrix sigmoid(const GPUMatrix &m) {
    GPUMatrix _m(m._row, m._col);
    sigmoid_interface(_m._data, m._data, m._row, m._col);
    return _m;
}

GPUMatrix d_sigmoid(const GPUMatrix &m) {
    return m * (m * (-1) + 1);
}

#endif //CUDAMLP_GPU_MATRIX_CUH
