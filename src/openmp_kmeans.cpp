/**
 * @file openmp_kmeans.cpp
 * @brief Implement sequential K-Means algorithm.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "openmp_kmeans.h"
#include <cmath>

bool check_equal_clusters1(std::vector<Point> dataset, std::vector<Point> old_dataset, int num_points) {
    for (auto i = 0; i < num_points; i++) {
        if (dataset[i].cluster_id != old_dataset[i].cluster_id) {
            return false;
        }
    }
    return true;
}

std::tuple<std::vector<Point>, std::vector<Point>>
openmp_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int num_clusters) {

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
#pragma omp parallel default(none) private(min_distance, distance, cluster_id) \
                                   shared(num_points, num_clusters, num_dimensions, dataset, centroids, count)
        {

#pragma omp for
            for (auto i = 0; i < num_points; i++) {
                min_distance = std::numeric_limits<double>::max();
                for (auto j = 0; j < num_clusters; j++) {
                    distance = 0;

                    // Euclidian distance.
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
            //std::fill (private_count.begin(),private_count.end(),0);
            std::vector<int> private_count(num_clusters, 0);

            // Initialize shared count.
#pragma omp single
            std::fill(count.begin(), count.end(), 0);

            // Initialize shared count.
#pragma omp for
            for (auto i = 0; i < num_clusters; i++) {
                std::fill(centroids[i].dimensions.begin(), centroids[i].dimensions.end(), 0);
            }

            std::vector<std::vector<double>> private_centroids(num_clusters, std::vector<double>(num_dimensions, 0));

#pragma omp for
            for (auto i = 0; i < num_points; i++) {
                for (auto j = 0; j < num_dimensions; j++) {
#pragma omp atomic
                    centroids[dataset[i].cluster_id].dimensions[j] += dataset[i].dimensions[j];
                    //private_centroids[dataset[i].cluster_id][j] += dataset[i].dimensions[j];
                }
#pragma omp atomic
                count[dataset[i].cluster_id]++;
                //private_count[dataset[i].cluster_id]++;
            }

#pragma omp for collapse(2)
            for (auto i = 0; i < num_clusters; i++) {
                for (auto j = 0; j < num_dimensions; j++) {
#pragma omp atomic
                    centroids[i].dimensions[j] /= count[i];
                }
            }
        }

        if (!first_iteration && check_equal_clusters1(dataset, old_dataset, num_points)) {
            convergence = true;
        } else {
            old_dataset = dataset;
            first_iteration = false;
        }
    }
    return {dataset, centroids};
}