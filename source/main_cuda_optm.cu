#include <iostream>
#include <chrono>
#include <vector>
#include "mlp_cuda_optm.cuh"

void test_MatrixOnGPU() {
    MatrixOnGPU m(3, 2);
    MatrixOnGPU n(3, 2);
    std::vector<MatrixOnGPU> t;
    t.push_back(m);

    std::cout << n%t.back().transpose() << std::endl;
}

void test_MatrixOnCPU() {
    MatrixOnCPU m = {
            {0., 1.},
            {2., 3.},
            {5., 4.},
    };
    std::cout << MatrixOnGPU(m) << std::endl;
}

void test_MLP_CUDA_Optm() {
    MLP_CUDA_Optm mlp({2, 16, 2});
    MatrixOnCPU x = {{2.},
                     {3.}};
    MatrixOnCPU y = {{0.},
                     {1.},};
//    std::cout << mlp << std::endl;
    mlp.back_propagation(x, y);
}

int main() {
    test_MatrixOnGPU();
//    test_MatrixOnCPU();
//    test_MLP_CUDA_Optm();
    return 0;
}