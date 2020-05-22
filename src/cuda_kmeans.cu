//
// Created by eliam on 22/05/2020.
//

#include "cuda_kmeans.cuh"



/**
    // Sequential version of K-Means Algorithms.

    @param dataset: dataset of points and assignments.
    @param num_clusters: total number of clusters to search.
    @param centroids: initial centroids of every clusters.
    @return dataset: final dataset of points and assignments.
    @return centroids: final centroids of every cluster.
*/
/*
std::tuple<std::vector<Point>, std::vector<Point>>
cuda_kmeans(std::vector<Point> dataset, const int num_clusters, std::vector<Point> centroids) {
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

*/