/**
 * @brief Header file for Sequential K-Means Algorithm.
 * @file sequential_kmeans.h
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PARALLEL_K_MEANS_SEQUENTIAL_KMEANS_H
#define PARALLEL_K_MEANS_SEQUENTIAL_KMEANS_H

#include "point.h"
#include <tuple>

bool check_convergence(const std::vector<Point> &dataset, const std::vector<Point> &old_dataset);

void update_centroids(const std::vector<Point> &dataset, int num_clusters, std::vector<Point> &centroids);

double calculate_distance(const std::vector<double> &first_point, const std::vector<double> &second_point);

void points_assignment(std::vector<Point> &dataset, int num_clusters, const std::vector<Point> &centroids);

std::tuple<std::vector<Point>, std::vector<Point>>
sequential_kmeans(std::vector<Point> dataset, int num_clusters, std::vector<Point> centroids);

#endif //PARALLEL_K_MEANS_SEQUENTIAL_KMEANS_H
