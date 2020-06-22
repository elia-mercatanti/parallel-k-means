/**
 * @brief Parallel K-Means version implemented with CUDA.
 * @file kmeans_cuda.cu
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "kmeans_cuda.cuh"
#include <cmath>
#include <iostream>

// Global variable in constant memory.
__constant__ int const_num_points, const_num_dimensions, const_num_clusters;

/**
    Check if an error is occurred, exit from the application in that case otherwise continue execution.

    @param file: file name where the error occurred.
    @param line: line of file where the error occurred.
    @param statement: statement of the line where the error occurred.
    @param err: error that occurred.
*/
__host__ void check_cuda_error(const char *file, const unsigned line, const char *statement, const cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":"
                  << line << "\n";
        exit(EXIT_FAILURE);
    }
}

/**
    Reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the
    same address. The operation is atomic. Method for double values.

    @param address: memory address on which to add the value.
    @param val: value to add to the memory address.
    @return old: return 64-bit word memory address.
*/
__device__ double doubleAtomicAdd(double *address, double val) {
    auto *address_as_ull =
            (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double((long long int) assumed)));
    } while (assumed != old);
    return __longlong_as_double((long long int) old);
}

/**
    Check if the convergence criterion has been reached, no change in assignments.

    @param host_assignments: array of point assignments to clusters.
    @param host_old_assignments: array of point assignments to clusters of the previous iteration.
    @param num_points: number of points into the dataset.
    @return boolean: return true if the assignments are the same, false otherwise.
*/
__host__ bool
check_convergence(const short *host_assignments, const short *host_old_assignments, const int num_points) {
    for (auto i = 0; i < num_points; i++) {
        if (host_assignments[i] != host_old_assignments[i]) {
            return false;
        }
    }
    return true;
}

/**
    Cuda Kernel that compute the new centroids of the iteration. Each thread manages a pair (centroid, dimension).

    @param device_centroids: array of centroids.
    @param device_num_points_clusters: array that stores the number of points in every cluster.
*/
__global__ void update_centroids(double *device_centroids, const int *device_num_points_clusters) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_clusters && col < const_num_dimensions) {
        device_centroids[row * const_num_dimensions + col] =
                device_centroids[row * const_num_dimensions + col] /
                (double(device_num_points_clusters[row]) / const_num_dimensions);
    }
}

/**
    Cuda Kernel that compute the sums of all points in a cluster, for every cluster. Each thread manages a pair
    (point, dimension).

    @param device_centroids: array of centroids.
    @param device_dataset: dataset of points.
    @param device_assignments: array of point assignments to clusters.
    @param device_num_points_clusters: array that stores the number of points in every cluster.
*/
__global__ void compute_sums(double *device_centroids, const double *device_dataset, const short *device_assignments,
                             int *device_num_points_clusters) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_points && col < const_num_dimensions) {
        short cluster_id = device_assignments[row];
        doubleAtomicAdd(&(device_centroids[cluster_id * const_num_dimensions + col]),
                        device_dataset[row * const_num_dimensions + col]);
        atomicAdd(&(device_num_points_clusters[cluster_id]), 1);
    }
}

/**
    Cuda Kernel that initialize every centroid to zero. Each thread manages a pair (centroid, dimension).

    @param device_centroids: array of centroids.
*/
__global__ void initialize_centroids(double *device_centroids) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_clusters && col < const_num_dimensions) {
        device_centroids[row * const_num_dimensions + col] = 0;
    }
}

/**
    Cuda Kernel that find the nearest centroid for all points, assign the point to that cluster. Each thread manages a
    single point.

    @param device_distances: array that stores all distances (points to clusters).
    @param device_assignments: array of point assignments to clusters.
*/
__global__ void points_assignment(const double *device_distances, short *device_assignments) {
    int unsigned thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto min_distance = INFINITY;
    double distance;
    short cluster_id;

    if (thread_id < const_num_points) {
        for (auto i = 0; i < const_num_clusters; i++) {
            distance = device_distances[thread_id * const_num_clusters + i];
            if (distance < min_distance) {
                min_distance = distance;
                cluster_id = i;
            }
        }
        device_assignments[thread_id] = cluster_id;
    }
}

/**
    Cuda Kernel that calculate all distance between points and clusters, use Euclidean distance. Each thread manages a
    pair (point, centroid).

    @param device_dataset: dataset of points.
    @param device_centroids: array of centroids.
    @param device_distances: array that stores all computed distances.
*/
__global__ void
compute_distances(const double *device_dataset, const double *device_centroids, double *device_distances) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    double distance = 0;

    if (row < const_num_points && col < const_num_clusters) {
        for (auto i = 0; i < const_num_dimensions; i++) {
            distance += pow(
                    device_dataset[row * const_num_dimensions + i] - device_centroids[col * const_num_dimensions + i],
                    2);
        }
        device_distances[row * const_num_clusters + col] = sqrt(distance);
    }
}

/**
    Parallel version of K-Means Algorithms using CUDA.

    @param device_dataset: dataset of points.
    @param num_clusters: total number of clusters to search.
    @param device_centroids: initial centroids of every clusters.
    @param num_points: number of points into the dataset.
    @param num_dimensions: number of dimensions of a point.
    @return host_assignments: final points assignments.
    @return device_centroids: final centroids of every cluster.
*/
__host__ std::tuple<short *, double *>
kmeans_cuda(const double *device_dataset, const short num_clusters, double *device_centroids, const int num_points,
            const short num_dimensions) {
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_num_clusters, &num_clusters, sizeof(short)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_num_dimensions, &num_dimensions, sizeof(short)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_num_points, &num_points, sizeof(int)));
    bool convergence = false;

    // Initialize dimension of grids and blocks for all the kernels.
    dim3 dim_block_distances(2, 512);
    dim3 dim_grid_distances(ceil(num_clusters / 2.0), ceil(num_points / 512.0));

    dim3 dim_block_initialize(16, 16);
    dim3 dim_grid_initialize(ceil(num_dimensions / 16.0), ceil(num_clusters / 16.0));

    dim3 dim_block_sums(2, 512);
    dim3 dim_grid_sums(ceil(num_dimensions / 2.0), ceil(num_points / 512.0));

    dim3 dim_block_centroids(16, 16);
    dim3 dim_grid_centroids(ceil(num_dimensions / 16.0), ceil(num_clusters / 16.0));

    // Initialize assignments arrays, distance array and count array.
    short *device_assignments;
    double *device_distances;
    int *device_num_points_clusters;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_assignments, num_points * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_distances, num_points * num_clusters * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_num_points_clusters, num_clusters * sizeof(int)));

    // Initialize host assignments array.
    auto *host_old_assignments = (short *) malloc(num_points * sizeof(short));
    auto *host_assignments = (short *) malloc(num_points * sizeof(short));

    do {
        // Assignment phase, compute all distances of any point from all clusters.
        compute_distances<<<dim_grid_distances, dim_block_distances>>>(device_dataset, device_centroids,
                                                                       device_distances);
        cudaDeviceSynchronize();

        // Assignment phase, find the nearest centroid for all points, assign the point to that cluster.
        points_assignment<<<ceil(num_points/1024.0), 1024>>>(device_distances, device_assignments);
        cudaDeviceSynchronize();

        // Update phase, initialize centroids.
        initialize_centroids<<<dim_grid_initialize, dim_block_initialize>>>(device_centroids);

        // Update phase, initialize counts, number of points in every cluster.
        CUDA_CHECK_RETURN(cudaMemset(device_num_points_clusters, 0, num_clusters * sizeof(int)));
        cudaDeviceSynchronize();

        // Update phase, compute sums of every points in all clusters.
        compute_sums<<<dim_grid_sums, dim_block_sums>>>(device_centroids, device_dataset, device_assignments,
                                                        device_num_points_clusters);
        cudaDeviceSynchronize();

        // Update phase, calculate mean of all points assigned to that cluster, for all cluster.
        update_centroids<<<dim_grid_centroids, dim_block_centroids>>>(device_centroids, device_num_points_clusters);
        cudaDeviceSynchronize();

        CUDA_CHECK_RETURN(cudaMemcpy(host_assignments, device_assignments, num_points * sizeof(short),
                                     cudaMemcpyDeviceToHost));

        // Check if the convergence criterion has been reached.
        if (check_convergence(host_assignments, host_old_assignments, num_points)) {
            convergence = true;
        } else {
            CUDA_CHECK_RETURN(cudaMemcpy(host_old_assignments, device_assignments, num_points * sizeof(short),
                                         cudaMemcpyDeviceToHost));
        }
    } while (!convergence);

    return {host_assignments, device_centroids};
}