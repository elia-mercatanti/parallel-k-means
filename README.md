# Parallel K-Means - Parallel Computing Project
Project for Parallel Computing course. Sequential and parallel implementations of K-Means algorithm in C++ with OpenMP 
and CUDA. The sequential K-Means version is compared with two parallel versions, tests include execution times and the 
speedup of each version.

More information can be found in the report of the project: [Report](https://github.com/elia-mercatanti/parallel-k-means/blob/master/report/report.pdf).

## Installation

1. Clone the repo.
```sh
git clone https://github.com/elia-mercatanti/parallel-k-means
```
2. Build with CMake.

## Usage

- For testing K-Means algorithm with all implementations to search clusters on a dataset, pass two arguments:
```bash 
parallel_kmeans <dataset file path> <number of clusters>
```
- For generating random datasets according to global variables, pass no arguments:
```bash 
parallel_kmeans <>
```

## Authors
* **Elia Mercatanti** - GitHub: [elia-mercatanti](https://github.com/elia-mercatanti)
* **Marco Calamai** - GitHub: [marcocalamai](https://github.com/marcocalamai)

## License
Licensed under the term of [MIT License](https://github.com/elia-mercatanti/parallel-k-means/blob/master/LICENSE).