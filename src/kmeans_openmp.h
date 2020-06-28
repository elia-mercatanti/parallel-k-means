/**
 * @brief Header file for Parallel K-Means Algorithm with OpenMP.
 * @file kmeans_openmp.h
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PARALLEL_K_MEANS_KMEANS_OPENMP_H
#define PARALLEL_K_MEANS_KMEANS_OPENMP_H

#include "point.h"
#include <tuple>

bool check_convergence_openmp(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset);

void update_centroids_openmp(const std::vector<Point> &dataset, int num_clusters, std::vector<Point> &centroids,
                             std::vector<int> &num_points_clusters);

double compute_distance_openmp(const std::vector<double> &first_point, const std::vector<double> &second_point);

void points_assignment_openmp(std::vector<Point> &dataset, int num_clusters, const std::vector<Point> &centroids);

std::tuple<std::vector<Point>, std::vector<Point>>
kmeans_openmp(std::vector<Point> dataset, int num_clusters, std::vector<Point> centroids);

#endif //PARALLEL_K_MEANS_KMEANS_OPENMP_H
