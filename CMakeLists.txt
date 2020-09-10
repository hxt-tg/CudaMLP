cmake_minimum_required(VERSION 3.17)
project(CudaMLP CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(MLP source/cpumatrix.cuh source/main.cu source/mlp.cuh source/cudamatrix.cuh source/random.cuh)
set_target_properties(
        MLP
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(MLP_CUDA_OPTM source/mlp_cuda_optm.cuh source/main_cuda_optm.cu)
target_link_libraries(MLP_CUDA_OPTM curand)

set_target_properties(
        MLP_CUDA_OPTM
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)