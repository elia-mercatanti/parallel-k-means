//
// Created by eliam on 18/04/2020.
//

#ifndef PC_PROJECT_POINT_H
#define PC_PROJECT_POINT_H

#include <vector>

// Structure for representing a Point of the dataset.
struct Point {
    std::vector<double> dimensions;
    int cluster_id{};
};

#endif //PC_PROJECT_POINT_H
