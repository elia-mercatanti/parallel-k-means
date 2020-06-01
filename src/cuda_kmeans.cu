/**
 * @file cuda_kmeans.cu
 * @brief Parallel K-Means version implemented with CUDA. First and Second phase parallel.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "cuda_kmeans.cuh"
#include <cmath>
#include <iostream>

__constant__ int const_num_points, const_num_dimensions, const_num_clusters;

void print_device(double *device, int row, int col) {
    double *host;
    host = (double *) malloc(row * col * sizeof(double));
    cudaMemcpy(host, device, row * col * sizeof(double), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout << "- " << host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void print_device(int *device, int row, int col) {
    int *host;
    host = (int *) malloc(row * col * sizeof(int));
    cudaMemcpy(host, device, row * col * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout << "- " << host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void print_device(short *device, int row, int col) {
    short *host;
    host = (short *) malloc(row * col * sizeof(short));
    cudaMemcpy(host, device, row * col * sizeof(short), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout << "- " << host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

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
    // Check if the convergence criterion has been reached, no change in assignments.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return convergence: return true of the assignments are the same.
*/
bool check_convergence(const short *host_assignments, const short *host_old_assignments, const int num_points) {
    for (auto i = 0; i < num_points; i++) {
        if (host_assignments[i] != host_old_assignments[i]) {
            return false;
        }
    }
    return true;
}

/**
    Update phase, calculate the mean of all points assigned to that cluster, for all cluster.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return void.
*/
__global__
void update_centroids(double *device_centroids, const int *device_count) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_clusters && col < const_num_dimensions) {
        device_centroids[row * const_num_dimensions + col] =
                device_centroids[row * const_num_dimensions + col] / (double(device_count[row]) / const_num_dimensions);
    }
}

/**
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return distance: Euclidean distance between first and second point.
*/
__global__
void compute_sums(double *device_centroids, const double *device_dataset, const short *device_assignments,
                  int *device_count) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_points && col < const_num_dimensions) {
        short cluster_id = device_assignments[row];
        doubleAtomicAdd(&(device_centroids[cluster_id * const_num_dimensions + col]),
                        device_dataset[row * const_num_dimensions + col]);
        atomicAdd(&(device_count[cluster_id]), 1);
    }
}

/**
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return distance: Euclidean distance between first and second point.
*/
__global__
void initialize_centroids(double *device_centroids) {
    int unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    int unsigned row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < const_num_clusters && col < const_num_dimensions) {
        device_centroids[row * const_num_dimensions + col] = 0;
    }
}

/**
    Assignment phase, find the nearest centroid for all points, assign the point to that cluster.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return void.
*/
__global__
void points_assignment(const double *device_distances, short *device_assignments) {
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
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return distance: Euclidean distance between first and second point.
*/
__global__
void compute_distance(const double *device_dataset, const double *device_centroids, double *device_distances) {
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
        //if (col == 2) {
        //    printf("col %d ", col);
        //}
        //if (device_distances[row * const_num_clusters + col]  == 0){
        //    printf("distanza %f ",device_distances[row * const_num_clusters + col] );
        //}
        //printf("distanza %f ", device_distances[2]);
        //printf("clusters %d ", const_num_clusters);
        //printf("clusters %d ", col);
    }
}

/*
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return distance: Euclidean distance between first and second point.

__global__
void initialize_assignments(short *device_assignments) {
    int unsigned thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id < const_num_points) {
        device_assignments[thread_id] = -1;
    }
}
*/

//Original compute sum with 2D grid (better with dataset with too much dimensions)
__global__
void
compute_sum2(const double *deviceDataset, double *deviceCentroids, const short *deviceAssignment, int *deviceCount) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < const_num_points) {
        short clusterId = deviceAssignment[row];
        for (auto i = 0; i < const_num_dimensions; i++) {
            doubleAtomicAdd(&deviceCentroids[clusterId * const_num_dimensions + i],
                            deviceDataset[row * const_num_dimensions + i]);
        }
        atomicAdd(&deviceCount[clusterId], 1);
    }
}

//Update centroids with 1D grid (no need to divide count for point's dimensions)
__global__
void update_centroids2(double *deviceCentroids, const int *deviceCount) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < const_num_dimensions && row < const_num_clusters) {
        deviceCentroids[row * const_num_dimensions + col] /= deviceCount[row];

    }
}

/**
    Parallel version of K-Means Algorithms using CUDA.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: initial centroids of every clusters.
    @return dataset: final dataset of points and assignments.
    @return centroids: final centroids of every cluster.
*/
std::tuple<short *, double *>
cuda_kmeans(double *device_dataset, const short num_clusters, double *device_centroids, const int num_points,
            const short num_dimensions) {
    cudaMemcpyToSymbol(const_num_clusters, &num_clusters, sizeof(short));
    cudaMemcpyToSymbol(const_num_dimensions, &num_dimensions, sizeof(short));
    cudaMemcpyToSymbol(const_num_points, &num_points, sizeof(int));
    bool convergence = false;
    short *device_old_assignments, *device_assignments, *host_old_assignments, *host_assignments;
    double *device_distances;
    int *device_count;
    int count = 0;

    // Initialize dim grid block.
    dim3 dim_block_distances(2, 512, 1);
    dim3 dim_grid_distances(ceil(num_clusters / 2.0), ceil(num_points / 512.0), 1);

    dim3 dim_block_initialize(32, 32, 1);
    dim3 dim_grid_initialize(ceil(num_dimensions / 32.0), ceil(num_clusters / 32.0), 1);

    dim3 dim_block_sums(2, 512, 1);
    dim3 dim_grid_sums(ceil(num_dimensions / 2.0), ceil(num_points / 512.0), 1);

    dim3 dim_block_centroids(32, 32, 1);
    dim3 dim_grid_centroids(ceil(num_dimensions / 32.0), ceil(num_clusters / 32.0), 1);

    // Initialize assignments arrays.
    cudaMalloc((void **) &device_old_assignments, num_points * sizeof(short));
    cudaMalloc((void **) &device_assignments, num_points * sizeof(short));
    //initialize_assignments<<<ceil(num_points/1024), 1024>>>(device_old_assignments);
    //cudaDeviceSynchronize();

    // Initialize distance array.
    cudaMalloc((void **) &device_distances, num_points * num_clusters * sizeof(double));

    // Initialize count array.
    cudaMalloc((void **) &device_count, num_clusters * sizeof(int));

    // Initialize host assignments array.
    host_old_assignments = (short *) malloc(num_points * sizeof(short));
    host_assignments = (short *) malloc(num_points * sizeof(short));

    while (!convergence) {

        // Assignment phase, compute all distances of any point from all clusters.
        compute_distance<<<dim_grid_distances, dim_block_distances>>>(device_dataset, device_centroids,
                                                                      device_distances);
        cudaDeviceSynchronize();

        // Assignment phase, find the nearest centroid for all points, assign the point to that cluster.
        points_assignment<<<ceil(num_points/1024.0), 1024>>>(device_distances, device_assignments);
        cudaDeviceSynchronize();

        // Update phase, initialize centroids.
        initialize_centroids<<<dim_grid_initialize, dim_block_initialize>>>(device_centroids);

        // Update phase, initialize counts, number of points in every cluster.
        cudaMemset(device_count, 0, num_clusters * sizeof(int));
        cudaDeviceSynchronize();

        // Update phase, compute sums of every points in all clusters.
        //compute_sums<<<dim_grid_sums, dim_block_sums>>>(device_centroids, device_dataset, device_assignments,
        //                                               device_count);
        compute_sum2<<<ceil(num_points/1024.0), 1024>>>(device_dataset, device_centroids, device_assignments,
                                                        device_count);
        cudaDeviceSynchronize();

        // Update phase, calculate mean of all points assigned to that cluster, for all cluster.
        //update_centroids<<<dim_grid_centroids, dim_block_centroids>>>(device_centroids, device_count);
        update_centroids2<<<dim_grid_centroids, dim_block_centroids>>>(device_centroids, device_count);
        cudaDeviceSynchronize();

        cudaMemcpy(host_assignments, device_assignments, num_points * sizeof(short),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(host_old_assignments, device_old_assignments, num_points * sizeof(short),
                   cudaMemcpyDeviceToHost);

        count++;

        // Check if the convergence criterion has been reached.
        if (check_convergence(host_assignments, host_old_assignments, num_points)) {
            convergence = true;
        } else {
            cudaMemcpy(device_old_assignments, device_assignments, num_points * sizeof(short),
                       cudaMemcpyDeviceToDevice);
        }
    }

    std::cout << "Numero di iterazioni: " << count << " \n";

    return {host_assignments, device_centroids};
}