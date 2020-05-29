//
// Created by eliam on 22/05/2020.
//

#ifndef PC_PROJECT_CUDA_KMEANS_CUH
#define PC_PROJECT_CUDA_KMEANS_CUH

#include "point.h"
#include <tuple>

std::tuple<short *, double *>
cuda_kmeans(double *device_dataset, short num_clusters, double *device_centroids, int num_points, short num_dimensions);

#endif //PC_PROJECT_CUDA_KMEANS_CUH
