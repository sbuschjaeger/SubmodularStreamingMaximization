#include <iostream>
#include <vector>
#include <math.h>

#include "functions/FastIVM.h"
#include "functions/kernels/RBFKernel.h"
#include "Greedy.h"
// #include "Random.h"
// #include "SieveStreaming.h"
#include "DataTypeHandling.h"
#include "functions/IVM.h"

// A RBF Kernel implementation
data_t kernel(const std::vector<data_t>& x1, const std::vector<data_t>& x2) {
    data_t distance = 0;
    if (x1 != x2) {
        for (unsigned int i = 0; i < x1.size(); ++i) {
            distance += (x1[i]-x2[i])*(x1[i]-x2[i]);
        }
        distance /= 1.0;
    }
    return 1.0 * std::exp(-distance);
}

// Compute the kernel matrix
inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X) {
    unsigned int K = X.size();
    Matrix mat(K);

    for (unsigned int i = 0; i < K; ++i) {
        for (unsigned int j = i; j < K; ++j) {
            data_t kval = kernel(X[i], X[j]);
            if (i == j) {
                mat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
            } else {
                mat(i,j) = kval / std::pow(1.0, 2.0);
                mat(j,i) = kval / std::pow(1.0, 2.0);
            }
        }
    }

    return mat;
}

int main() {
    // TODO: Add more test cases

    // "Generate" some test data
    std::vector<std::vector<data_t>> data = {
        {0.0, 0.0},
        {1.0, 1.0},
        {0.5, 1.0},
        {1.0, 0.5},
        {0.0, 0.5},
        {0.0, 1.5},
        {0.0, 1.0},
        {0.5, 0.5},
    };    

    // "Generate" some unique ids for each data item
    std::vector<idx_t> ids = {
        1, 2, 3, 4, 5, 6, 7, 8
    };

    // First lets check if we compute the correct kernel and its logdet via a cholesky decomposition. 
    // 
    Matrix m = compute_kernel(data);
    std::cout << "Matrix is: " << std::endl;
    std::cout << to_string(m) << std::endl;
    
    Matrix L = cholesky(m);
    std::cout << "Regular L: " << std::endl;
    std::cout << to_string(L) << std::endl;
    std::cout << "log det from chol is: " << log_det_from_cholesky(L) << std::endl;
    std::cout << "log det directly is: " << log_det(m) << std::endl << std::endl;

    // Lets select a summary of size K = 5
    unsigned int K = 5;

    // Two different IVM variants
    // IVM slowIVM(RBFKernel(), 1.0);
    FastIVM fastIVM(K, RBFKernel(), 1.0);
    
    // Use the Greedy algorithm
    Greedy greedy(K, fastIVM);
    greedy.fit(data, ids);
    std::vector<std::vector<double>> solution = greedy.get_solution();
    std::vector<idx_t> solution_ids = greedy.get_ids();
    double fval = greedy.get_fval();

    std::cout << "Found a solution with fval = " << fval << std::endl;
    for (auto x : solution) {
        for (auto xi : x) {
            std::cout << xi << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "ids = {";
    for (auto i : solution_ids) {
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

    // Use the Random algorithm
    Random random(K, fastIVM);
    random.fit(data, ids);
    std::vector<std::vector<double>> solution = random.get_solution();
    std::vector<idx_t> solution_ids = random.get_ids();
    double fval = random.get_fval();

    std::cout << "Found a solution with fval = " << fval << std::endl;
    for (auto x : solution) {
        for (auto xi : x) {
            std::cout << xi << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "ids = {";
    for (auto i : solution_ids) {
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

    // Use the SieveStreaming algorithm, but now with a lambda function instead of a fastIVM object
    SieveStreaming sieve(K, [](std::vector<std::vector<data_t>> const &X){
        unsigned int K = X.size();
        data_t * kmat = new data_t[K*K];

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel(X[i], X[j]);
                if (i == j) {
                    kmat[i+i*K] = 1.0 + kval / std::pow(1.0, 2.0);
                } else {
                    kmat[i+j*K] = kval / std::pow(1.0, 2.0);
                    kmat[j+i*K] = kval / std::pow(1.0, 2.0);
                }
            }
        }

        data_t fval = logDet(kmat, X.size(), X.size());
        delete [] kmat;
        return fval;
    }, 2.0, 0.1);
    sieve.fit(data);
    std::vector<std::vector<double>> solution = sieve.get_solution();
    std::vector<idx_t> solution_ids = sieve.get_ids();
    double fval = sieve.get_fval();

    std::cout << "Found a solution with fval = " << fval << std::endl;
    for (auto x : solution) {
        for (auto xi : x) {
            std::cout << xi << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "ids = {";
    for (auto i : solution_ids) {
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;
    return 0;
}