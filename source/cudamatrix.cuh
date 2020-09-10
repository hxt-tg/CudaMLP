#ifndef CUDAMLP_CUDAMATRIX_CUH
#define CUDAMLP_CUDAMATRIX_CUH

#include <iostream>
#include <iomanip>
#include <exception>
#include <random>
#include <array>
#include <cuda.h>
#include <curand.h>

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
 * Shape of a: row x mid
 * Shape of b: mid x col
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

class CUDAMatrix {
    friend std::ostream &operator<<(std::ostream &out, const CUDAMatrix &m) {
        out << "CUDAMatrix of " << m.row << " x " << m.col;
        return out;
    }

    static float *malloc_matrix(const unsigned &r, const unsigned &c) {
        auto m = new float[r * c];
        return m;
    }

public:
    CUDAMatrix() = default;

    CUDAMatrix(const unsigned &r, const unsigned &c, const float &val = 0.0) : CUDAMatrix() {
        if (!r || !c) return;
        row = r;
        col = c;
        size = row * col * sizeof(float);
        CudaCheck(cudaMalloc(&data, size));
        initialize_interface(data, val, row, col);
    }

    CUDAMatrix(const CUDAMatrix &m) : CUDAMatrix(m.row, m.col) {
        CudaCheck(cudaMemcpy(data, m.data, size, cudaMemcpyDeviceToDevice));
    }

    explicit CUDAMatrix(const CPUMatrix &m) : CUDAMatrix() {
        if (!m.row || !m.col) return;
        row = m.row;
        col = m.col;
        size = row * col * sizeof(float);
        CudaCheck(cudaMalloc(&data, size));
        for (auto i = 0; i < row; i++)
            CudaCheck(cudaMemcpy(data + i * col, m.data[i],
                                 col * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~CUDAMatrix() {
        clear();
    }

    void clear() {
        if (!size) return;
        CudaCheck(cudaFree(data));
        row = col = size = 0;
        data = nullptr;
    }

    // Operators
    CUDAMatrix &operator=(const CUDAMatrix &m) {
        if (&m == this) return *this;
        if (row && col)
            clear();

        row = m.row;
        col = m.col;
        size = m.size;
        CudaCheck(cudaMalloc(&data, size));
        CudaCheck(cudaMemcpy(data, m.data, size, cudaMemcpyDeviceToDevice));
        return *this;
    }

    CUDAMatrix &operator=(const std::initializer_list<std::initializer_list<float>> &arr) {
        CUDAMatrix _m(arr);
        return (*this = _m);
    }

    CUDAMatrix operator+(const CUDAMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CUDAMatrix _m(row, col);
        element_wise_interface(_m.data, data, m.data, row, col, KERNEL_ADD);
        return _m;
    }

    CUDAMatrix operator-(const CUDAMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CUDAMatrix _m(row, col);
        element_wise_interface(_m.data, data, m.data, row, col, KERNEL_SUB);
        return _m;
    }

    CUDAMatrix operator*(const CUDAMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CUDAMatrix _m(row, col);
        element_wise_interface(_m.data, data, m.data, row, col, KERNEL_MUL);
        return _m;
    }

    CUDAMatrix operator/(const CUDAMatrix &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        CUDAMatrix _m(row, col);
        element_wise_interface(_m.data, data, m.data, row, col, KERNEL_DIV);
        return _m;
    }

    CUDAMatrix operator+(const float &scalar) const {
        CUDAMatrix _m(row, col);
        scalar_interface(_m.data, data, scalar, row, col, KERNEL_ADD);
        return _m;
    }

    CUDAMatrix operator-(const float &scalar) const {
        CUDAMatrix _m(row, col);
        scalar_interface(_m.data, data, scalar, row, col, KERNEL_SUB);
        return _m;
    }

    CUDAMatrix operator*(const float &scalar) const {
        CUDAMatrix _m(row, col);
        scalar_interface(_m.data, data, scalar, row, col, KERNEL_MUL);
        return _m;
    }

    CUDAMatrix operator/(const float &scalar) const {
        CUDAMatrix _m(row, col);
        scalar_interface(_m.data, data, scalar, row, col, KERNEL_DIV);
        return _m;
    }

    CUDAMatrix operator%(const CUDAMatrix &m) const {
        if (col != m.row)
            throw std::runtime_error("Shape is not compatible. Column should equal to row.");
        CUDAMatrix _m(row, m.col);
        matmul_interface(_m.data, data, m.data, row, col, m.col);
        return _m;
    }

    CUDAMatrix &operator+=(const CUDAMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(data, data, m.data, row, col, KERNEL_ADD);
        return (*this);
    }

    CUDAMatrix &operator-=(const CUDAMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(data, data, m.data, row, col, KERNEL_SUB);
        return (*this);
    }

    CUDAMatrix &operator*=(const CUDAMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(data, data, m.data, row, col, KERNEL_MUL);
        return (*this);
    }

    CUDAMatrix &operator/=(const CUDAMatrix &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_interface(data, data, m.data, row, col, KERNEL_DIV);
        return (*this);
    }

    CUDAMatrix &operator+=(const float &scalar) {
        scalar_interface(data, data, scalar, row, col, KERNEL_ADD);
        return (*this);
    }

    CUDAMatrix &operator-=(const float &scalar) {
        scalar_interface(data, data, scalar, row, col, KERNEL_SUB);
        return (*this);
    }

    CUDAMatrix &operator*=(const float &scalar) {
        scalar_interface(data, data, scalar, row, col, KERNEL_MUL);
        return (*this);
    }

    CUDAMatrix &operator/=(const float &scalar) {
        scalar_interface(data, data, scalar, row, col, KERNEL_DIV);
        return (*this);
    }

    CUDAMatrix &operator%=(const CUDAMatrix &m) {
        CUDAMatrix n = (*this) % m;
        (*this) = n;
        return (*this);
    }

    CUDAMatrix transpose() const {
        unsigned new_row = col, new_col = row;
        CUDAMatrix _m(new_row, new_col);
        transpose_interface(_m.data, data, new_row, new_col);
        return _m;
    }

    float sum() const {
        auto d = new float[size];
        CudaCheck(cudaMemcpy(d, data, size, cudaMemcpyDeviceToHost));

        float s = 0;
        for (auto i = 0; i < row * col; i++)
            s += d[i];
        delete[] d;
        return s;
    }

    void print_all_numbers(bool endl = true) const {
        auto d = new float[size];
        CudaCheck(cudaMemcpy(d, data, size, cudaMemcpyDeviceToHost));

        for (auto i = 0; i < row; ++i) {
            for (auto j = 0; j < col; ++j)
                std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                          << d[i * col + j];
            std::cout << std::endl;
        }
        if (endl) std::cout << std::endl;
        delete[] d;
    }

public:
    unsigned row{0};
    unsigned col{0};
    size_t size{0};
    float *data{nullptr};
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

CUDAMatrix sigmoid(const CUDAMatrix &m) {
    CUDAMatrix _m(m.row, m.col);
    sigmoid_interface(_m.data, m.data, m.row, m.col);
    return _m;
}

CUDAMatrix d_sigmoid(const CUDAMatrix &m) {
    return m * (m * (-1) + 1);
}

#endif //CUDAMLP_CUDAMATRIX_CUH
