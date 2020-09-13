#include <iostream>
#include <chrono>
#include "mlp/cpu_matrix.cuh"
#include "mlp/gpu_matrix.cuh"
#include "mlp/random.cuh"
#include "mlp/mlp.cuh"

void test_CUDAMatrix() {
    GPUMatrix a(CPUMatrix({{0., 1.},
                           {2., 3.},
                           {5., 4.}}));
    GPUMatrix b(random_normal(2, 5));
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
void test_MLP(unsigned iter_times = 100000, const char *label="MLP") {
    MLP<MatrixClass> mlp({786, 16, 16, 10});
    std::cout << mlp << std::endl;
    std::vector<MatrixClass> data_x, data_y;
    for (auto i = 0; i < 100; i++) {
        data_x.push_back(MatrixClass(random_normal(786, 1)));
        data_y.push_back(MatrixClass(random_normal(10, 1)));
    }

    auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < iter_times; i++)
        if (i % 10000 == 0)
            std::cout << "\r" << i << ": " << mlp.learn(data_x, data_y)
                      << (i == 0 ? "\n" : "");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "[" << label << "] Training time: " << duration.count() << " ms" << std::endl;
}

int main() {
    test_MLP<CPUMatrix>(1000000, "CPU-MLP");
    test_MLP<GPUMatrix>(1000000, "GPU-MLP");
    return 0;
}
