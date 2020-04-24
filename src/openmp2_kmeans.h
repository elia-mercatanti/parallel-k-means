//
// Created by eliam on 23/04/2020.
//

#ifndef PC_PROJECT_OPENMP2_KMEANS_H
#define PC_PROJECT_OPENMP2_KMEANS_H

#include "point.h"
#include <tuple>

bool check_equal_clusters3(std::vector<Point> dataset, std::vector<Point> old_dataset, int num_points);

std::tuple<std::vector<Point>, std::vector<Point>>
openmp2_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int num_clusters);

#endif //PC_PROJECT_OPENMP2_KMEANS_H
