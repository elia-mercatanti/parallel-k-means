/**
 * @brief Header file for the struct Point used to represents points and theirs dimensions.
 * @file point.h
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PARALLEL_K_MEANS_POINT_H
#define PARALLEL_K_MEANS_POINT_H

#include <vector>

// Structure for representing a Point of the dataset.
struct Point {
    std::vector<double> dimensions;
    int cluster_id;
};

#endif //PARALLEL_K_MEANS_POINT_H
