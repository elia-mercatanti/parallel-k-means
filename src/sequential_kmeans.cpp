/**
 * @file sequential_kmeans.cpp
 * @brief Implement Sequential K-Means Algorithm.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "sequential_kmeans.h"
#include <cmath>

bool check_equal_clusters(std::vector<Point> dataset, std::vector<Point> old_dataset, int num_points) {
    for (auto i = 0; i < num_points; i++) {
        if (dataset[i].cluster_id != old_dataset[i].cluster_id) {
            return false;
        }
    }
    return true;
}

std::tuple<std::vector<Point>, std::vector<Point>>
sequential_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int num_clusters) {

    // Sequential K-Means algorithm.
    const auto num_points = dataset.size();
    const auto num_dimensions = dataset[0].dimensions.size();
    bool convergence = false, first_iteration = true;
    std::vector<Point> old_dataset;
    std::vector<int> count(num_clusters);
    double min_distance, distance;
    int cluster_id;

    while (!convergence) {

        // Assignment phase, find the nearest centroid, assign the Point to that cluster.
        for (auto i = 0; i < num_points; i++) {
            min_distance = std::numeric_limits<double>::max();
            for (auto j = 0; j < num_clusters; j++) {
                distance = 0;
                for (auto k = 0; k < num_dimensions; k++) {
                    distance += pow(dataset[i].dimensions[k] - centroids[j].dimensions[k], 2);
                }
                distance = sqrt(distance);

                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_id = j;
                }
            }
            dataset[i].cluster_id = cluster_id;
        }

        // Update phase, find the nearest centroid, assign the Point to that cluster.
        std::fill(count.begin(), count.end(), 0);
        for (auto i = 0; i < num_clusters; i++) {
            std::fill(centroids[i].dimensions.begin(), centroids[i].dimensions.end(), 0);
        }
        for (auto i = 0; i < num_points; i++) {
            for (auto j = 0; j < num_dimensions; j++) {
                centroids[dataset[i].cluster_id].dimensions[j] += dataset[i].dimensions[j];
            }
            count[dataset[i].cluster_id]++;
        }
        for (auto i = 0; i < num_clusters; i++) {
            for (auto j = 0; j < num_dimensions; j++) {
                centroids[i].dimensions[j] /= count[i];
            }
        }

        if (!first_iteration && check_equal_clusters(dataset, old_dataset, num_points)) {
            convergence = true;
        } else {
            old_dataset = dataset;
            first_iteration = false;
        }
    }
    return {dataset, centroids};
}