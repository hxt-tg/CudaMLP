#include <iostream>
#include <chrono>
#include "cpumatrix.cuh"
#include "cudamatrix.cuh"
#include "random.cuh"
#include "mlp.cuh"

void test_CUDAMatrix() {
    CUDAMatrix a(CPUMatrix({{0., 1.},
                            {2., 3.},
                            {5., 4.}}));
    CUDAMatrix b(random_matrix(2, 5));
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << a % b << std::endl;
    std::cout << a.transpose() << std::endl;
    a.print_all_numbers();
    a += a;
    a.print_all_numbers();
    sigmoid(a).print_all_numbers();
    std::cout << a * 3 << std::endl;
}

template <class MatrixClass=CPUMatrix>
void test_MLP(unsigned iter_times = 100000) {
    MLP<MatrixClass> mlp({786, 256, 256, 10});
    std::cout << mlp << std::endl;
    std::vector<MatrixClass> data_x, data_y;
    for (auto i = 0; i < 100; i++) {
        data_x.push_back(MatrixClass(random_matrix(786, 1)));
        data_y.push_back(MatrixClass(random_matrix(10, 1)));
    }

    for (auto i = 0; i < iter_times; i++)
        if (i % 1000000 == 0)
            std::cout << "\r" << i << ": " << mlp.learn(data_x, data_y)
                      << (i == 0 ? "\n" : "");
}

int main() {
    auto begin = std::chrono::high_resolution_clock::now();
    test_MLP<CPUMatrix>();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "[CPU-MLP] Training time: " << duration.count() << " ms" << std::endl;

    begin = std::chrono::high_resolution_clock::now();
    test_MLP<CUDAMatrix>();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "[CUDA-MLP] Training time: " << duration.count() << " ms" << std::endl;

    return 0;
}
