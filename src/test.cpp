/**
 * @brief Main file where Sequential K-Means is compared with two Parallel versions implemented with OpenMP and CUDA.
 *        This file test the execution times and the speedup of each version.
 * @file test.cpp
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "point.h"
#include "sequential_kmeans.h"
#include "kmeans_openmp.h"
#include "kmeans_cuda.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <numeric>
#include <random>
#include <chrono>
#include <algorithm>

// Variables for the creation of random datasets, number of points and dimensions are provided foreach datasets.
int num_points_rand[] = {1000, 10000, 100000, 1000000};
int num_dimensions_rand[] = {10, 100, 1000};

/**
    Generate random datasets according to global variables (Values from 0.0 to 1.0).
*/
void generate_random_datasets() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto num_points: num_points_rand) {
        for (auto num_dimensions: num_dimensions_rand) {
            const std::string rand_dataset_name =
                    "rand_dataset_" + std::to_string(num_points) + "x" + std::to_string(num_dimensions) +
                    ".txt";

            // Create random dataset only if is not already generated.
            const std::string rand_dataset_path = "../datasets/" + rand_dataset_name;
            if (!(std::ifstream(rand_dataset_path))) {
                std::ofstream rand_dataset_file(rand_dataset_path);
                for (auto i = 0; i < num_points; i++) {
                    for (auto j = 0; j < num_dimensions; j++) {
                        rand_dataset_file << distribution(generator) << " ";
                    }
                    rand_dataset_file << "\n";
                }
                rand_dataset_file.close();

                std::cout << "Random Dataset Generated: " + rand_dataset_name + "\n";
            }
        }
    }
}

/**
    Read and load the dataset to be tested from his file.

    @param dataset: vector structure where the dataset is saved.
    @param dataset_file: file of the dataset.
*/
void load_dataset(std::vector<Point> &dataset, std::ifstream &dataset_file) {
    std::cout << "Reading in progress ...\n";
    std::string file_line;

    // Get every points of the dataset.
    while (getline(dataset_file, file_line)) {
        std::istringstream string_stream(file_line);
        Point point;
        double dimension_value;

        // Get every dimensions of the point.
        while (string_stream >> dimension_value) {
            point.dimensions.push_back(dimension_value);
        }
        dataset.push_back(point);
    }
}

/**
    Generate initial centroids chosen randomly from the dataset.

    @param dataset: dataset of points.
    @param num_clusters: number of clusters.
    @return initial_centroids: initial centroids chosen randomly from the dataset.
*/
std::vector<Point> generate_initial_centroids(const std::vector<Point> &dataset, const long num_clusters) {
    std::vector<Point> initial_centroids(num_clusters);
    std::vector<int> random_vector(dataset.size());
    std::iota(random_vector.begin(), random_vector.end(), 0);
    std::shuffle(random_vector.begin(), random_vector.end(), std::mt19937(std::random_device()()));
    for (auto i = 0; i < num_clusters; i++) {
        initial_centroids[i] = dataset[random_vector[i]];
        initial_centroids[i].cluster_id = i;
    }
    return initial_centroids;
}

/**
    // Transform a vector of Points in a simple array.

    @param data: vector of Points to be Transformed.
    @param num_rows: number of rows for data.
    @param num_columns: number of columns for data.
    @param array: simple array where the data is moved.
*/
void
transform_into_array(const std::vector<Point> &data, const int num_rows, const int num_columns, double *array) {
    for (auto i = 0; i < num_rows; i++) {
        for (auto j = 0; j < num_columns; j++) {
            array[i * num_columns + j] = data[i].dimensions[j];
        }
    }
}

int main(int argc, char *argv[]) {

    // Check the number of parameters.
    if (argc != 1 && argc != 3) {
        std::cerr
                << "Usage:\n"
                << "- parallel_kmeans.exe <dataset file path> <number of clusters> : "
                << "For testing K-Means algorithm with various implementations.\n"
                << "- parallel_kmeans.exe <> : For generating random datasets according to global variables.\n";
        exit(EXIT_FAILURE);
    }

    // If no arguments are provided generate random datasets according to global variables (Values from 0 to 1).
    if (argc == 1) {
        std::cout << "- Random Datasets Generation -\n\n";
        generate_random_datasets();
        std::cout << "\nRandom Datasets Generation Done\n";
        exit(EXIT_SUCCESS);
    }

    std::cout << "< Parallel K-Means - Testing OpenMP and CUDA implementation of the K-Means algorithm >\n\n";

    // Get number of clusters to search.
    const auto num_clusters = std::strtol(argv[2], nullptr, 0);
    if (num_clusters == 0) {
        std::cerr << "Error: Could not obtain the number of clusters, you have inserted: " << argv[2] << "\n";
        exit(EXIT_FAILURE);
    }

    // Dataset of points.
    std::vector<Point> dataset;

    // Check the existence of the dataset file and get dataset points from the input file.
    std::ifstream dataset_file(argv[1]);
    if (dataset_file) {
        std::cout << "- Dataset Load -\n\n";
        load_dataset(dataset, dataset_file);
        dataset_file.close();
        std::cout << "Loading Done\n\n";
    } else {
        std::cerr << "Error: Could not open file: " << argv[1] << "\n";
        exit(EXIT_FAILURE);
    }

    // Generate num_clusters random initial centroids.

    std::cout << "- Generating Initial Centroids -\n\n";
    std::vector<Point> initial_centroids = generate_initial_centroids(dataset, num_clusters);
    std::cout << "Done\n\n";

    std::cout << "- Testing K-Means Algorithm -\n\n";

    // Results of K-Means.
    std::vector<Point> final_dataset, final_centroids;

    // Measure execution time of Sequential K-Means.
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = sequential_kmeans(dataset, num_clusters, initial_centroids);
    auto finish = std::chrono::high_resolution_clock::now();
    const double sequential_execution_time = std::chrono::duration<double>(finish - start).count();

    std::cout << "- Sequential K-Means:\n";
    std::cout << "Execution Time: " << sequential_execution_time << " s\n\n";

    // Measure execution time and speedup of Parallel K-Means with OpenMP.
    start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = kmeans_openmp(dataset, num_clusters, initial_centroids);
    finish = std::chrono::high_resolution_clock::now();
    double parallel_execution_time = std::chrono::duration<double>(finish - start).count();

    std::cout << "- Parallel K-Means with OpenMP:\n";
    std::cout << "Execution Time: " << parallel_execution_time << " s\n";
    std::cout << "Speedup: " << sequential_execution_time / parallel_execution_time << "\n\n";

    // Create host arrays for CUDA implementation, including new assignments array.
    const auto num_points = dataset.size();
    const auto num_dimensions = dataset[0].dimensions.size();
    const auto num_bytes_dataset = num_points * num_dimensions * sizeof(double);
    const auto num_bytes_centroids = num_clusters * num_dimensions * sizeof(double);
    auto *host_dataset = (double *) malloc(num_bytes_dataset);
    auto *host_centroids = (double *) malloc(num_bytes_centroids);
    auto *host_assignments = (short *) malloc(num_points * sizeof(short));

    // Dataset and centroids transformation in a simple array.
    transform_into_array(dataset, num_points, num_dimensions, host_dataset);
    transform_into_array(initial_centroids, num_clusters, num_dimensions, host_centroids);

    // Loads arrays into the device using CUDA.
    double *device_dataset, *device_centroids;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_dataset, num_bytes_dataset));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_centroids, num_bytes_centroids));
    CUDA_CHECK_RETURN(cudaMemcpy(device_dataset, host_dataset, num_bytes_dataset, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(device_centroids, host_centroids, num_bytes_centroids, cudaMemcpyHostToDevice));

    // Measure execution time and speedup of Parallel K-Means with CUDA.
    start = std::chrono::high_resolution_clock::now();
    std::tie(host_assignments, device_centroids) = kmeans_cuda(device_dataset, num_clusters, device_centroids,
                                                               num_points, num_dimensions);
    finish = std::chrono::high_resolution_clock::now();
    parallel_execution_time = std::chrono::duration<double>(finish - start).count();

    std::cout << "- Parallel K-Means with CUDA:\n";
    std::cout << "Execution Time: " << parallel_execution_time << " s\n";
    std::cout << "Speedup: " << sequential_execution_time / parallel_execution_time << "\n";

    CUDA_CHECK_RETURN(cudaFree(device_dataset));
    CUDA_CHECK_RETURN(cudaFree(device_centroids));

    free(host_dataset);
    free(host_centroids);
    free(host_assignments);

    return 0;
}