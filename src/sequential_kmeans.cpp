/**
 * @file sequential_kmeans.cpp
 * @brief Implement Sequential K-Means Algorithm.
 * @authors Elia Mercatanti, Marco Calamai
*/

#include "sequential_kmeans.h"
#include <cmath>

/**
    // Check if the convergence criterion has been reached, no change in assignments.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: centroids of every cluster.
    @return convergence: return true of the assignments are the same.
*/
bool check_convergence(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset) {
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
void update_centroids(const std::vector<Point> &dataset, const int num_clusters, std::vector<Point> &centroids) {
    const auto num_dimensions = dataset[0].dimensions.size();
    std::vector<int> num_points_clusters(num_clusters, 0);

    // Reset the centroids.
    for (auto i = 0; i < num_clusters; i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            centroids[i].dimensions[j] = 0;
        }
    }

    // Calculate the sums and total number of points in each cluster.
    for (auto i = 0; i < dataset.size(); i++) {
        for (auto j = 0; j < num_dimensions; j++) {
            centroids[dataset[i].cluster_id].dimensions[j] += dataset[i].dimensions[j];
        }
        num_points_clusters[dataset[i].cluster_id]++;
    }

    // Calculate the new centroids, for all cluster.
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
    @return distance: Euclidean distance between first and second point.
*/
double calculate_distance(const std::vector<double> &first_point, const std::vector<double> &second_point) {
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
void points_assignment(std::vector<Point> &dataset, const int num_clusters, const std::vector<Point> &centroids) {
    double min_distance, distance;
    int cluster_id;

    for (auto i = 0; i < dataset.size(); i++) {
        min_distance = std::numeric_limits<double>::max();
        for (auto j = 0; j < num_clusters; j++) {
            distance = calculate_distance(dataset[i].dimensions, centroids[j].dimensions);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_id = centroids[j].cluster_id;
            }
        }
        dataset[i].cluster_id = cluster_id;
    }
}

/**
    // Sequential version of K-Means Algorithms.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: initial centroids of every clusters.
    @return dataset: final dataset of points and assignments.
    @return centroids: final centroids of every cluster.
*/
std::tuple<std::vector<Point>, std::vector<Point>>
sequential_kmeans(std::vector<Point> dataset, const int num_clusters, std::vector<Point> centroids) {
    bool convergence = false;
    std::vector<Point> old_dataset = dataset;

    do {
        // Assignment phase, find the nearest centroid for all points, assign the point to that cluster.
        points_assignment(dataset, num_clusters, centroids);

        // Update phase, calculate mean of all points assigned to that cluster, for all cluster.
        update_centroids(dataset, num_clusters, centroids);

        // Check if the convergence criterion has been reached.
        if (check_convergence(dataset, old_dataset)) {
            convergence = true;
        } else {
            old_dataset = dataset;
        }
    } while (!convergence);

    return {dataset, centroids};
}