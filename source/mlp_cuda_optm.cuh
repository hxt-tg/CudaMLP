#ifndef CUDAMLP_MLP_CUDA_OPTM_CUH
#define CUDAMLP_MLP_CUDA_OPTM_CUH

#include <iostream>
#include <vector>
#include <exception>
#include <cmath>
#include <cuda.h>
#include <curand.h>

#define BLOCK_SIZE  32
#define RAND_SEED   8726452ULL

/* ======== CUDA/CURAND check ======== */
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

class MatrixOnCPU {
    friend std::ostream &operator<<(std::ostream &out, const MatrixOnCPU &m) {
        out << "MatrixOnCPU of " << m.row << " x " << m.col;
        return out;
    }

public:
    unsigned row{0};
    unsigned col{0};
    size_t size{0};
    float *data{nullptr};

    MatrixOnCPU() = default;

    MatrixOnCPU(const unsigned &r, const unsigned &c) : MatrixOnCPU() {
        if (!r || !c) return;
        row = r;
        col = c;
        size = row * col * sizeof(float);
        data = new float[size];
    }

    MatrixOnCPU(const std::initializer_list<std::initializer_list<float>> &arr) : MatrixOnCPU() {
        if (arr.size() == 0 || arr.begin()->size() == 0) return;

        auto arr_n_rows = arr.size(), arr_n_cols = arr.begin()->size();
        for (auto it = arr.begin(); it != arr.end(); it++)
            if (it->size() != arr_n_cols)
                throw std::runtime_error("Not a matrix.");

        row = arr_n_rows;
        col = arr_n_cols;
        size = row * col * sizeof(float);
        data = new float[size];
        unsigned i = 0;
        for (auto row_it = arr.begin(); row_it < arr.end(); i++, row_it++) {
            for (auto col_it = row_it->begin(); col_it < row_it->end(); i++, col_it++)
                data[i] = *col_it;
        }
    }

    ~MatrixOnCPU() {
        if (!size) return;
        delete data;
        row = col = size = 0;
        data = nullptr;
    }
};

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

void element_wise_device(float *result, const float *a, const float *b, unsigned row, unsigned col,
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

void scalar_device(float *result, const float *a, const float b, unsigned row, unsigned col,
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

void matmul_device(float *result, const float *a, const float *b,
                   unsigned row, unsigned mid, unsigned col) {
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

void transpose_device(float *result, const float *a, unsigned row, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, row, col);
    CudaCheck(cudaDeviceSynchronize());
}

/* ======== Sigmoid kernels ======== */
__global__ void sigmoid_kernel(float *result, const float *a, unsigned row, unsigned col) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= row || j >= col) return;
    result[i * col + j] = 1 / (1 + exp(-a[i * col + j]));
}

void sigmoid_device(float *result, const float *a, unsigned row, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, row, col);
    CudaCheck(cudaDeviceSynchronize());
}

__global__ void d_sigmoid_kernel(float *result, const float *a, unsigned row, unsigned col) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= row || j >= col) return;
    result[i * col + j] = a[i * col + j] * (1 - a[i * col + j]);
}

void d_sigmoid_device(float *result, const float *a, unsigned row, unsigned col) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((row + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (col + threadsPerBlock.y - 1) / threadsPerBlock.y);
    d_sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, row, col);
    CudaCheck(cudaDeviceSynchronize());
}


class MatrixOnGPU {
    friend std::ostream &operator<<(std::ostream &out, const MatrixOnGPU &m) {
        out << "MatrixOnGPU of " << m.row << " x " << m.col;
        return out;
    }

public:
    unsigned row{0};
    unsigned col{0};
    size_t size{0};
    float *data{nullptr};

    MatrixOnGPU() = default;

    MatrixOnGPU(const unsigned &r, const unsigned &c) : MatrixOnGPU() {
        if (!r || !c) return;
        row = r;
        col = c;
        size = r * c * sizeof(float);
        CudaCheck(cudaMalloc(&data, size));
    }

    MatrixOnGPU(const MatrixOnGPU &m) : MatrixOnGPU() {
        if (!m.size) return;
        row = m.row;
        col = m.col;
        size = m.size;
        CudaCheck(cudaMalloc(&data, size));
        CudaCheck(cudaMemcpy(data, m.data, size, cudaMemcpyDeviceToDevice));
    }

    explicit MatrixOnGPU(const MatrixOnCPU &m) : MatrixOnGPU() {
        if (!m.size) return;
        row = m.row;
        col = m.col;
        size = m.size;
        CudaCheck(cudaMalloc(&data, size));
        data_from_cpu(m);
    }

    ~MatrixOnGPU() {
        clear();
    }
    
    void data_from_cpu(const MatrixOnCPU &m) const {
        if (size != m.size) throw std::runtime_error("Size of data does not match.");
        CudaCheck(cudaMemcpy(data, m.data, size, cudaMemcpyHostToDevice));
    }

    void data_to_cpu(const MatrixOnCPU &m) const {
        if (size != m.size) throw std::runtime_error("Size of data does not match.");
        CudaCheck(cudaMemcpy(m.data, data, size, cudaMemcpyDeviceToHost));
    }
    
    void clear() {
        if (!size) return;
        CudaCheck(cudaFree(data));
        row = col = size = 0;
        data = nullptr;
    }
    
    // Operators
    MatrixOnGPU &operator=(const MatrixOnGPU &m) {
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

    MatrixOnGPU operator+(const MatrixOnGPU &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        MatrixOnGPU _m(row, col);
        element_wise_device(_m.data, data, m.data, row, col, KERNEL_ADD);
        return _m;
    }

    MatrixOnGPU operator-(const MatrixOnGPU &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        MatrixOnGPU _m(row, col);
        element_wise_device(_m.data, data, m.data, row, col, KERNEL_SUB);
        return _m;
    }

    MatrixOnGPU operator*(const MatrixOnGPU &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        MatrixOnGPU _m(row, col);
        element_wise_device(_m.data, data, m.data, row, col, KERNEL_MUL);
        return _m;
    }

    MatrixOnGPU operator/(const MatrixOnGPU &m) const {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        MatrixOnGPU _m(row, col);
        element_wise_device(_m.data, data, m.data, row, col, KERNEL_DIV);
        return _m;
    }

    MatrixOnGPU operator+(const float &scalar) const {
        MatrixOnGPU _m(row, col);
        scalar_device(_m.data, data, scalar, row, col, KERNEL_ADD);
        return _m;
    }

    MatrixOnGPU operator-(const float &scalar) const {
        MatrixOnGPU _m(row, col);
        scalar_device(_m.data, data, scalar, row, col, KERNEL_SUB);
        return _m;
    }

    MatrixOnGPU operator*(const float &scalar) const {
        MatrixOnGPU _m(row, col);
        scalar_device(_m.data, data, scalar, row, col, KERNEL_MUL);
        return _m;
    }

    MatrixOnGPU operator/(const float &scalar) const {
        MatrixOnGPU _m(row, col);
        scalar_device(_m.data, data, scalar, row, col, KERNEL_DIV);
        return _m;
    }

    MatrixOnGPU operator%(const MatrixOnGPU &m) const {
        if (col != m.row)
            throw std::runtime_error("Shape is not compatible. Column should equal to row.");
        MatrixOnGPU _m(row, m.col);
        matmul_device(_m.data, data, m.data, row, col, m.col);
        return _m;
    }

    MatrixOnGPU &operator+=(const MatrixOnGPU &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_device(data, data, m.data, row, col, KERNEL_ADD);
        return (*this);
    }

    MatrixOnGPU &operator-=(const MatrixOnGPU &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_device(data, data, m.data, row, col, KERNEL_SUB);
        return (*this);
    }

    MatrixOnGPU &operator*=(const MatrixOnGPU &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_device(data, data, m.data, row, col, KERNEL_MUL);
        return (*this);
    }

    MatrixOnGPU &operator/=(const MatrixOnGPU &m) {
        if (col != m.col || row != m.row)
            throw std::runtime_error("Shape is not compatible.");
        element_wise_device(data, data, m.data, row, col, KERNEL_DIV);
        return (*this);
    }

    MatrixOnGPU &operator+=(const float &scalar) {
        scalar_device(data, data, scalar, row, col, KERNEL_ADD);
        return (*this);
    }

    MatrixOnGPU &operator-=(const float &scalar) {
        scalar_device(data, data, scalar, row, col, KERNEL_SUB);
        return (*this);
    }

    MatrixOnGPU &operator*=(const float &scalar) {
        scalar_device(data, data, scalar, row, col, KERNEL_MUL);
        return (*this);
    }

    MatrixOnGPU &operator/=(const float &scalar) {
        scalar_device(data, data, scalar, row, col, KERNEL_DIV);
        return (*this);
    }

    MatrixOnGPU &operator%=(const MatrixOnGPU &m) {
        MatrixOnGPU n = (*this) % m;
        (*this) = n;
        return (*this);
    }

    MatrixOnGPU transpose() {
        unsigned new_row = col, new_col = row;
        MatrixOnGPU _m(new_row, new_col);
        transpose_device(_m.data, data, new_row, new_col);
        return _m;
    }

};

MatrixOnGPU sigmoid(const MatrixOnGPU &a) {
    MatrixOnGPU result(a.row, a.col);
    sigmoid_device(result.data, a.data, a.row, a.col);
    return result;
}

MatrixOnGPU d_sigmoid(const MatrixOnGPU &a) {
    MatrixOnGPU result(a.row, a.col);
    d_sigmoid_device(result.data, a.data, a.row, a.col);
    return result;
}

class MLP_CUDA_Optm {
    friend std::ostream &operator<<(std::ostream &out, const MLP_CUDA_Optm &m) {
        if (m.n_layers == 0)
            out << "Empty MLP (CUDA Optimized)";
        else {
            out << m.n_layers << "-layer MLP (CUDA Optimized) [dim=(" << m.dim[0];
            for (auto i = 1; i < m.dim.size(); i++)
                out << ", " << m.dim[i];
            out << ")  learn_rate=" << m.eta << "]";
        }
        return out;
    }

public:
    MLP_CUDA_Optm() = default;

    explicit MLP_CUDA_Optm(const std::vector<unsigned> &layer_dim, const float learn_rate = 0.1) {
        dim = layer_dim;
        eta = learn_rate;
        n_layers = layer_dim.size() - 1;
        if (n_layers < 1)
            throw std::runtime_error("There should be at least 2 layers.");

        curandGenerator_t gen;
        CurandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CurandCheck(curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED));

        for (auto i = 0; i < n_layers; i++) {
            auto *wp = new MatrixOnGPU(layer_dim[i + 1], layer_dim[i]);
            auto *tp = new MatrixOnGPU(layer_dim[i + 1], 1);
            CurandCheck(curandGenerateNormal(gen, wp->data, wp->size, 0, 1));
            CurandCheck(curandGenerateNormal(gen, tp->data, tp->size, 0, 1));
            w.push_back(wp);
            t.push_back(tp);
        }
        CurandCheck(curandDestroyGenerator(gen));
    }

    ~MLP_CUDA_Optm() {
        if (!n_layers) return;
        for (auto &wp : w) delete wp;
        for (auto &tp : t) delete tp;
        n_layers = 0;
    }

    float back_propagation(const MatrixOnCPU &x, const MatrixOnCPU &y) {
//        std::vector<MatrixOnGPU *> b;
//        std::vector<MatrixOnGPU *> g(n_layers + 1, nullptr);
//        std::vector<MatrixOnGPU *> delta_w(n_layers + 1, nullptr);
//        std::vector<MatrixOnGPU *> delta_t(n_layers + 1, nullptr);
//        b.push_back(new MatrixOnGPU(x));

//        for (auto i = 0; i < n_layers; i++) {
//            auto _b = new MatrixOnGPU(t[i]->row, t[i]->col);
//            matmul(*_b, *w[i], *_b);
//            *_b += *t[i];
//            sigmoid(*_b, *_b);
//            b.push_back(_b);
//        }
//
//        // compute b in each layer
//        auto _g = new MatrixOnGPU(*b.back());
//        auto _y = new MatrixOnGPU(y);
//        *_y -= *_g;
//        d_sigmoid(*_g, *_g);
//        *_g *= *_y;
//        g[g.size()-1] = _g;

        return 0.;

    }

//    MatrixClass feed_forward(const MatrixClass &x) const {
//        MatrixClass _x(x);
//        for (auto i = 0; i < n_layers; i++)
//            _x = sigmoid(w[i] % _x + t[i]);
//        return _x;
//    }
//
//    float back_propagation(const MatrixClass &x, const MatrixClass &y) {
//        std::vector<MatrixClass> b = {x};
//        std::vector<MatrixClass> g(n_layers + 1, MatrixClass(1, 1));
//        std::vector<MatrixClass> delta_w(n_layers + 1, MatrixClass());
//        std::vector<MatrixClass> delta_t(n_layers + 1, MatrixClass());
//        MatrixClass _x(x);
//        // compute b in each layer
//        for (auto i = 0; i < n_layers; i++) {
//            _x = sigmoid(w[i] % _x + t[i]);
//            b.push_back(_x);
//        }
//
//        // compute the gradient, delta_w and delta_theta of output layer
//        g[g.size() - 1] = d_sigmoid(b[b.size() - 1]) * (y - b[b.size() - 1]);
//        delta_w[delta_w.size() - 1] = (g[g.size() - 1] % b[b.size() - 2].transpose()) * eta;
//        delta_t[delta_t.size() - 1] = g[g.size() - 1] * (-eta);
//
//        // compute the gradient, delta_w, delta_theta of hidden layer
//        for (auto i = n_layers - 1; i > 0; i--) {
//            g[i] = d_sigmoid(b[i]) * (w[i].transpose() % g[i + 1]);
//            delta_w[i] = (g[i] % b[i - 1].transpose()) * (eta);
//            delta_t[i] = g[i] * (-eta);
//        }
//
//        // update w and theta of each layer
//        for (auto i = 0; i < n_layers; i++) {
//            w[i] += delta_w[i + 1];
//            t[i] += delta_t[i + 1];
//        }
//
//        MatrixClass mse(y - b[b.size() - 1]);
//        return (mse * mse).sum();
//    }
//
//    float learn(const std::vector<MatrixClass> &data_x, const std::vector<MatrixClass> &data_y) {
//        float total_error = 0;
//        for (auto i = 0; i < data_x.size(); i++)
//            total_error += 0.5f * back_propagation(data_x[i], data_y[i]);
//        return total_error;
//    }
//
//    std::vector<MatrixClass> predict(const std::vector<MatrixClass> &data_x) const {
//        std::vector<MatrixClass> predict_y;
//        for (auto &x : data_x)
//            predict_y.push_back(feed_forward(x));
//        return predict_y;
//    }

private:
    std::vector<unsigned> dim{};
    unsigned n_layers{0};
    float eta{0.0};
    std::vector<MatrixOnGPU *> w{};
    std::vector<MatrixOnGPU *> t{};
};

#endif //CUDAMLP_MLP_CUDA_OPTM_CUH
