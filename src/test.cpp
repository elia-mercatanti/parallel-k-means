#include "point.h"
#include "sequential_kmeans.h"
#include "openmp_kmeans.h"
#include <iostream>
#include <vector>
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

    // Dataset of points and total number of dimensions.
    std::vector<Point> dataset;

    // Check the existence of the dataset file and get dataset points from the input file.
    std::ifstream dataset_file(argv[1]);
    if (dataset_file) {
        std::string line;
        double dimension_value;

        // Get every points in the dataset.
        while (getline(dataset_file, line)) {
            std::istringstream ss(line);
            Point my_point;
            while (ss >> dimension_value) {
                my_point.dimensions.push_back(dimension_value);
            }
            dataset.push_back(my_point);
        }
        dataset_file.close();
    } else {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the total number of points in the dataset, total dimensions of points and number of clusters.
    const auto num_points = dataset.size();
    const auto num_dimensions = dataset[0].dimensions.size();
    const auto num_clusters = std::strtol(argv[2], nullptr, 0);
    if (num_clusters == 0) {
        std::cerr << "Could not obtain the number of clusters, you have inserted: " << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Generate num_clusters random centroids.
    std::vector<Point> centroids(num_clusters);
    std::vector<int> random_vector(num_points);
    std::iota(random_vector.begin(), random_vector.end(), 0);
    std::shuffle(random_vector.begin(), random_vector.end(), std::mt19937(std::random_device()()));
    for (auto i = 0; i < num_clusters; i++) {
        centroids[i] = dataset[random_vector[i]];
        centroids[i].cluster_id = i;
    }

    std::vector<Point> result_dataset, result_centroids;

    // Measure execution time of Sequential K-Means.
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(result_dataset, result_centroids) = sequential_kmeans(dataset, centroids, num_clusters);
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Sequential K-Means Execution Time: " << elapsed.count() << " s\n";

    // Print vector points
    for (auto point : result_centroids) {
        for (auto j = 0; j < num_dimensions; j++) {
            std::cout << point.dimensions[j] << " ";
        }
        std::cout << std::endl;
    }

    // Measure execution time of Parallel K-Means with OpenMP.
    start = std::chrono::high_resolution_clock::now();
    std::tie(result_dataset, result_centroids) = openmp_kmeans(dataset, centroids, num_clusters);
    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish - start;
    std::cout << "Parallel K-Means With OpenMP Execution Time: " << elapsed.count() << " s\n";

    // Print vector points
    for (auto point : result_centroids) {
        for (auto j = 0; j < num_dimensions; j++) {
            std::cout << point.dimensions[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}