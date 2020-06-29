/**
 * @brief Parallel K-Means version implemented with OpenMP.
 * @file kmeans_openmp.cpp
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "kmeans_openmp.h"
#include <cmath>

/**
    Check if the convergence criterion has been reached, no change in assignments.

    @param dataset: dataset of points and assignments.
    @param old_dataset: dataset of points and assignments of the previous iteration.
    @return boolean: return true if the assignments are the same, false otherwise.
*/
bool check_convergence_openmp(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset) {
    for (auto i = 0; i < dataset.size(); i++) {
        if (dataset[i].cluster_id != old_dataset[i].cluster_id) {
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
    @param num_points_clusters: array that stores the number of points in every cluster.
*/
void update_centroids_openmp(const std::vector<Point> &dataset, const int num_clusters, std::vector<Point> &centroids,
                             std::vector<int> &num_points_clusters) {
    const auto num_dimensions = dataset[0].dimensions.size();

    // Reset shared num_points_clusters.
#pragma omp for
    for (auto i = 0; i < num_clusters; i++) {
        num_points_clusters[i] = 0;
    }

    // Reset the centroids.
#pragma omp for collapse(2)
    for (auto i = 0; i < num_clusters; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            centroids[i].dimensions[j] = 0;
        }
    }

    // Calculate the sums and total number of points in each cluster.
#pragma omp for
    for (auto i = 0; i < dataset.size(); i++) {
        for (auto j = 0; j < num_dimensions; j++) {
#pragma omp atomic
            centroids[dataset[i].cluster_id].dimensions[j] += dataset[i].dimensions[j];
        }
#pragma omp atomic
        num_points_clusters[dataset[i].cluster_id]++;
    }

    // Calculate the new centroids, for all cluster.
#pragma omp for collapse(2)
    for (auto i = 0; i < num_clusters; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            centroids[i].dimensions[j] /= num_points_clusters[i];
        }
    }
}

/**
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return boolean: Euclidean distance between first and second point.
*/
double compute_distance_openmp(const std::vector<double> &first_point, const std::vector<double> &second_point) {
    double distance = 0;
    for (auto i = 0; i < first_point.size(); i++) {
        distance += pow(first_point[i] - second_point[i], 2);
    }
    return sqrt(distance);
}

/**
    Assignment phase, find the nearest centroid for all points, assign the point to that cluster.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
*/
void
points_assignment_openmp(std::vector<Point> &dataset, const int num_clusters, const std::vector<Point> &centroids) {
    double min_distance, distance;
    int cluster_id;

#pragma omp for
    for (auto i = 0; i < dataset.size(); i++) {
        min_distance = std::numeric_limits<double>::max();
        for (auto j = 0; j < num_clusters; j++) {
            distance = compute_distance_openmp(dataset[i].dimensions, centroids[j].dimensions);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_id = centroids[j].cluster_id;
            }
        }
        dataset[i].cluster_id = cluster_id;
    }
}

/**
    Parallel version of K-Means Algorithms implemented with OpenMP.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: initial centroids of every clusters.
    @return dataset: final dataset of points and assignments.
    @return centroids: final centroids of every cluster.
*/
std::tuple<std::vector<Point>, std::vector<Point>>
kmeans_openmp(std::vector<Point> dataset, const int num_clusters, std::vector<Point> centroids) {
    bool convergence = false;
    std::vector<Point> old_dataset = dataset;
    std::vector<int> num_points_clusters(num_clusters);

    do {
#pragma omp parallel num_threads(4) default(none) shared(dataset, num_clusters, centroids, num_points_clusters)
        {
            // Assignment phase, find the nearest centroid for all points, assign the point to that cluster.
            points_assignment_openmp(dataset, num_clusters, centroids);

            // Update phase, calculate mean of all points assigned to that cluster, for all cluster.
            update_centroids_openmp(dataset, num_clusters, centroids, num_points_clusters);
        }
        if (check_convergence_openmp(dataset, old_dataset)) {
            convergence = true;
        } else {
            old_dataset = dataset;
        }
    } while (!convergence);

    return {dataset, centroids};
}