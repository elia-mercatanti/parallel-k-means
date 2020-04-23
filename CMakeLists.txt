cmake_minimum_required(VERSION 3.15)
project(pc_project)

set(CMAKE_CXX_STANDARD 20)

add_executable(pc_project src/sequential_kmeans.cpp src/openmp_kmeans.cpp src/openmp_kmeans.h
        src/sequential_kmeans.h src/test.cpp src/point.h)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(pc_project PUBLIC OpenMP::OpenMP_CXX)
endif ()