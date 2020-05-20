/**
 * @file openmp_kmeans.h
 * @brief Header file for Parallel K-Means Algorithm with OpenMP, first version.
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PC_PROJECT_OPENMP_KMEANS_H
#define PC_PROJECT_OPENMP_KMEANS_H

#include "point.h"
#include <tuple>

bool check_convergence_openmp1(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset);

void update_centroids_openmp1(const std::vector<Point> &dataset, int num_clusters, std::vector<Point> &centroids,
                              std::vector<int> &num_points_clusters);

double calculate_distance_openmp1(const std::vector<double> &first_point, const std::vector<double> &second_point);

void points_assignment_openmp1(std::vector<Point> &dataset, int num_clusters, const std::vector<Point> &centroids);

std::tuple<std::vector<Point>, std::vector<Point>>
openmp_kmeans1(std::vector<Point> dataset, int num_clusters, std::vector<Point> centroids);

#endif //PC_PROJECT_OPENMP_KMEANS_H
