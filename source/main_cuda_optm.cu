#include <iostream>
#include <chrono>
#include <vector>
#include "random.cuh"
#include "cpumatrix.cuh"
#include "cudamatrix.cuh"

void test_CUDAMatrix() {
    CUDAMatrix a(CPUMatrix({
            {0., 1.},
            {2., 3.},
            {5., 4.},
    }));
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


int main() {
    test_CUDAMatrix();
//    test_MLP_CUDA_Optm();
    return 0;
}