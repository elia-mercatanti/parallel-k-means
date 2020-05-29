/**
 * @file test.cpp
 * @brief Main file where Sequential K-Means is compared with Parallel versions implemented with OpenMP and CUDA.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "point.h"
#include "sequential_kmeans.h"
#include "openmp_kmeans.h"
#include "openmp2_kmeans.h"
#include "cuda_kmeans.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <numeric>
#include <random>
#include <chrono>
#include <algorithm>

int main(int argc, char *argv[]) {
    // Check the number of parameters.
    if (argc != 3) {
        std::cerr << "Usage: pc_project.exe <data-file-path> <n. clusters>" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get number of clusters to search.
    const auto num_clusters = std::strtol(argv[2], nullptr, 0);
    if (num_clusters == 0) {
        std::cerr << "Could not obtain the number of clusters, you have inserted: " << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Dataset of points.
    std::vector<Point> dataset;

    // Check the existence of the dataset file and get dataset points from the input file.
    std::ifstream dataset_file(argv[1]);
    if (dataset_file) {
        std::string file_line;
        double dimension_value;
        int points_read = 0;

        std::cout << "- Dataset Load -\n " << std::endl;
        std::cout << "Reading in progress ..." << std::endl;

        // Get every points in the dataset.
        while (getline(dataset_file, file_line)) {
            std::istringstream string_stream(file_line);
            Point point;

            while (string_stream >> dimension_value) {
                point.dimensions.push_back(dimension_value);
            }
            dataset.push_back(point);
            points_read++;

        }
        dataset_file.close();
        std::cout << "Loading Done\n" << std::endl;
    } else {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Generate num_clusters random initial centroids.
    std::vector<Point> initial_centroids(num_clusters);
    std::vector<int> random_vector(dataset.size());
    std::iota(random_vector.begin(), random_vector.end(), 0);
    std::shuffle(random_vector.begin(), random_vector.end(), std::mt19937(std::random_device()()));
    for (auto i = 0; i < num_clusters; i++) {
        initial_centroids[i] = dataset[random_vector[i]];
        initial_centroids[i].cluster_id = i;
    }

    /*
    // Print centroids
    for (const auto &centroid : initial_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
     */

    // Results of K-Means.
    std::vector<Point> final_dataset, final_centroids;

    std::cout << "- Testing K-Means Clustering -\n" << std::endl;

    // Measure execution time of Sequential K-Means.
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = sequential_kmeans(dataset, num_clusters, initial_centroids);
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Sequential K-Means Execution Time: " << elapsed.count() << " s\n" << std::endl;

    /*
    // Print centroids
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    */

    // Measure execution time of Parallel K-Means with OpenMP.
    start = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::tie(final_dataset, final_centroids) = openmp_kmeans1(dataset, num_clusters, initial_centroids);
    std::cout << std::endl;
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With OpenMP (First Version) Execution Time: " << elapsed.count() << " s\n"
              << std::endl;

    /*
    // Print centroids.
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
     */

    // Measure execution time of Parallel K-Means with OpenMP2.
    start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = openmp_kmeans2(dataset, num_clusters, initial_centroids);
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With OpenMP (Second Version) Execution Time: " << elapsed.count() << " s\n"
              << std::endl;

    /*
    // Print centroids
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
     */

    // CUDA
    auto num_points = dataset.size();
    auto num_dimensions = dataset[0].dimensions.size();
    auto num_bytes_dataset = num_points * num_dimensions * sizeof(double);
    auto num_bytes_centroids = num_clusters * num_dimensions * sizeof(double);
    double *host_dataset, *host_centroids, *device_dataset, *device_centroids;
    short *host_assignments;

    host_dataset = (double *) malloc(num_bytes_dataset);
    host_centroids = (double *) malloc(num_bytes_centroids);
    host_assignments = (short *) malloc(num_points * sizeof(short));

    // Dataset transformation.
    for (auto i = 0; i < num_points; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            host_dataset[i * num_dimensions + j] = dataset[i].dimensions[j];
        }
    }

    // Centroids transformation.
    for (auto i = 0; i < num_clusters; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            host_centroids[i * num_dimensions + j] = initial_centroids[i].dimensions[j];
        }
    }

    // Load arrays into the device.
    cudaMalloc((void **) &device_dataset, num_bytes_dataset);
    cudaMalloc((void **) &device_centroids, num_bytes_centroids);
    cudaMemcpy(device_dataset, host_dataset, num_bytes_dataset, cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, host_centroids, num_bytes_centroids, cudaMemcpyHostToDevice);

    // Measure execution time of Parallel K-Means with CUDA.
    start = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::tie(host_assignments, device_centroids) = cuda_kmeans(device_dataset, num_clusters, device_centroids,
                                                               num_points, num_dimensions);
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With CUDA Execution Time: " << elapsed.count() << std::endl;

    cudaMemcpy(host_centroids, device_centroids, num_bytes_centroids, cudaMemcpyDeviceToHost);

    /*
    // Centroids transformation.
    for (auto i = 0; i < num_clusters; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            std::cout << host_centroids[i * num_dimensions + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_dataset);
    cudaFree(device_centroids);
    free(host_dataset);
    free(host_centroids);
    free(host_assignments);

    return 0;
}