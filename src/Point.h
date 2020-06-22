/**
 * @brief Header file for the struct Point used to represents points and theirs dimensions.
 * @file point.h
 * @authors Elia Mercatanti, Marco Calamai
*/

#ifndef PC_PROJECT_POINT_H
#define PC_PROJECT_POINT_H

#include <vector>

// Structure for representing a Point of the dataset.
struct Point {
    std::vector<double> dimensions;
    int cluster_id;
};

#endif //PC_PROJECT_POINT_H
