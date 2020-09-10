cmake_minimum_required(VERSION 3.17)
project(CudaMLP CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaMLP source/cpumatrix.cuh source/main.cu source/mlp.cuh source/cudamatrix.cuh source/random.cuh)
set_target_properties(
        CudaMLP
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
