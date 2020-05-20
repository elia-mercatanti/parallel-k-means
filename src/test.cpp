/**
 * @file test.cpp
 * @brief Main file where Sequential K-Means is compared with Parallel versions implemented with OpenMP and CUDA.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "point.h"
#include "sequential_kmeans.h"
#include "openmp_kmeans.h"
#include "openmp2_kmeans.h"
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

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

        std::cout << "- Dataset Loading -\n " << std::endl;

        // Get every points in the dataset.
        while (getline(dataset_file, file_line)) {
            std::istringstream string_stream(file_line);
            Point point;

            while (string_stream >> dimension_value) {
                point.dimensions.push_back(dimension_value);
            }
            dataset.push_back(point);
            points_read++;
            std::cout << "\rPoints Read: " << points_read << std::flush;
        }
        dataset_file.close();
        std::cout << "\n" << std::endl;
    } else {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the total number of points in the dataset and total dimensions for points.
    const auto num_points = dataset.size();
    const auto num_dimensions = dataset[0].dimensions.size();

    // Generate num_clusters random initial centroids.
    std::vector<Point> initial_centroids(num_clusters);
    std::vector<int> random_vector(num_points);
    std::iota(random_vector.begin(), random_vector.end(), 0);
    std::shuffle(random_vector.begin(), random_vector.end(), std::mt19937(std::random_device()()));
    for (auto i = 0; i < num_clusters; i++) {
        initial_centroids[i] = dataset[random_vector[i]];
        initial_centroids[i].cluster_id = i;
    }

    // Results of K-Means.
    std::vector<Point> final_dataset, final_centroids;

    std::cout << "- Testing K-Means Clustering -\n" << std::endl;

    // Measure execution time of Sequential K-Means.
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = sequential_kmeans(dataset, num_clusters, initial_centroids);
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Sequential K-Means Execution Time: " << elapsed.count() << " s\n" << std::endl;

    // Print centroids
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Measure execution time of Parallel K-Means with OpenMP.
    start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = openmp_kmeans1(dataset, num_clusters, initial_centroids);
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With OpenMP (First Version) Execution Time: " << elapsed.count() << " s\n"
              << std::endl;

    // Print centroids.
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Measure execution time of Parallel K-Means with OpenMP2.
    start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = openmp_kmeans2(dataset, num_clusters, initial_centroids);
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With OpenMP (Second Version) Execution Time: " << elapsed.count() << " s\n"
              << std::endl;

    // Print centroids
    for (const auto &centroid : final_centroids) {
        for (auto value : centroid.dimensions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}