cmake_minimum_required(VERSION 3.15)
project(parallel_kmeans LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 20)

add_executable(parallel_kmeans src/sequential_kmeans.cpp src/kmeans_openmp.cpp src/kmeans_openmp.h
        src/sequential_kmeans.h src/test.cpp src/point.h src/kmeans_cuda.cu src/kmeans_cuda.cuh)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(parallel_kmeans PUBLIC OpenMP::OpenMP_CXX)
endif ()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
