//
// Created by eliam on 18/04/2020.
//

#ifndef PC_PROJECT_SEQUENTIAL_KMEANS_H
#define PC_PROJECT_SEQUENTIAL_KMEANS_H

#include "point.h"
#include <tuple>

bool check_equal_clusters(std::vector<Point> dataset, std::vector<Point> old_dataset, int num_points);

std::tuple<std::vector<Point>, std::vector<Point>>
sequential_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int num_clusters);

#endif //PC_PROJECT_SEQUENTIAL_KMEANS_H
