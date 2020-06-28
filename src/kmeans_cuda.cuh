/**
 * @brief Header file for Parallel K-Means Algorithm with CUDA.
 * @file kmeans_cuda.h
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PARALLEL_K_MEANS_KMEANS_CUDA_CUH
#define PARALLEL_K_MEANS_KMEANS_CUDA_CUH

#include "cuda_runtime.h"
#include "point.h"
#include <tuple>

#define CUDA_CHECK_RETURN(value) check_cuda_error(__FILE__, __LINE__, #value, value)

__host__ void check_cuda_error(const char *file, unsigned line, const char *statement, cudaError_t err);

__device__ double doubleAtomicAdd(double *address, double val);

__host__ bool check_convergence(const short *host_assignments, const short *host_old_assignments, int num_points);

__global__ void update_centroids(double *device_centroids, const int *device_count);

__global__ void compute_sums(double *device_centroids, const double *device_dataset, const short *device_assignments,
                             int *device_count);

__global__ void initialize_centroids(double *device_centroids);

__global__ void points_assignment(const double *device_distances, short *device_assignments);

__global__ void
compute_distances(const double *device_dataset, const double *device_centroids, double *device_distances);

__host__ std::tuple<short *, double *>
kmeans_cuda(const double *device_dataset, short num_clusters, double *device_centroids, int num_points,
            short num_dimensions);

#endif //PARALLEL_K_MEANS_KMEANS_CUDA_CUH
