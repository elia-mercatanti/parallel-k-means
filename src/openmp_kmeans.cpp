/**
 * @file openmp_kmeans.cpp
 * @brief Parallel K-Means version implemented with OpenMP. First and Second phase parallel.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "openmp_kmeans.h"
#include <cmath>

/**
    // Check if the convergence criterion has been reached, no change in assignments.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return convergence: return true of the assignments are the same.
*/
bool check_convergence_openmp1(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset) {
    for (auto i = 0; i < dataset.size(); i++) {
        if (dataset[i].cluster_id != old_dataset[i].cluster_id) {
            return false;
        }
    }
    return true;
}

/**
    // Update phase, calculate the mean of all points assigned to that cluster, for all cluster.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return void.
*/
void update_centroids_openmp1(const std::vector<Point> &dataset, const int num_clusters, std::vector<Point> &centroids,
                              std::vector<int> &num_points_clusters) {
    const auto num_dimensions = dataset[0].dimensions.size();

    // Initialize shared num_points_clusters.
#pragma omp single
    std::fill(num_points_clusters.begin(), num_points_clusters.end(), 0);

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
#pragma omp atomic
            centroids[i].dimensions[j] /= num_points_clusters[i];
        }
    }
}

/**
    Calculate distance between two points, use Euclidean distance.

    @param first_point: first point to be compared.
    @param second_point: second point to be compared.
    @return distance: Euclidean distance between first and second point.
*/
double calculate_distance_openmp1(const std::vector<double> &first_point, const std::vector<double> &second_point) {
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
    @return void.
*/
void
points_assignment_openmp1(std::vector<Point> &dataset, const int num_clusters, const std::vector<Point> &centroids) {
    double min_distance, distance;
    int cluster_id;

#pragma omp for
    for (auto i = 0; i < dataset.size(); i++) {
        min_distance = std::numeric_limits<double>::max();
        for (auto j = 0; j < num_clusters; j++) {
            distance = calculate_distance_openmp1(dataset[i].dimensions, centroids[j].dimensions);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_id = centroids[j].cluster_id;
            }
        }
        dataset[i].cluster_id = cluster_id;
    }
}

/**
    // Parallel version of K-Means Algorithms implemented with OpenMP. First and Second phase parallel.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: initial centroids of every clusters.
    @return dataset: final dataset of points and assignments.
    @return centroids: final centroids of every cluster.
*/
std::tuple<std::vector<Point>, std::vector<Point>>
openmp_kmeans1(std::vector<Point> dataset, const int num_clusters, std::vector<Point> centroids) {
    bool convergence = false;
    std::vector<Point> old_dataset = dataset;
    std::vector<int> num_points_clusters(num_clusters);

    do {
#pragma omp parallel default(none) shared(dataset, num_clusters, centroids, num_points_clusters)
        {
            // Assignment phase, find the nearest centroid for all points, assign the point to that cluster.
            points_assignment_openmp1(dataset, num_clusters, centroids);

            // Update phase, calculate mean of all points assigned to that cluster, for all cluster.
            update_centroids_openmp1(dataset, num_clusters, centroids, num_points_clusters);
        }
        if (check_convergence_openmp1(dataset, old_dataset)) {
            convergence = true;
        } else {
            old_dataset = dataset;
        }
    } while (!convergence);

    return {dataset, centroids};
}