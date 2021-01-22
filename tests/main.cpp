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

    std::vector<std::vector<data_t>> data = {
        {0.0, 0.0},
        {1.0, 1.0},
        {0.5, 1.0},
        {1.0, 0.5},

        {0.0, 0.5},
        {0.0, 1.5},
        {0.0, 1.0},
        {0.5, 0.5},
        
        // {2.0, 0.0}
        // {1.0, 2.0},
        // {1.0, 1.5},
        // {2.0, 1.0},

        // {2.0, 0.5},
        // {0.25, 1.0},
        // {1.0, 1.25},
        // {2.25, 1.0}
    };    

    std::vector<idx_t> ids = {
        1, 2, 3, 4, 5, 6, 7, 8
    };

    unsigned int K = 5;
    Matrix m = compute_kernel(data);
    // std::cout << "Matrix is: " << std::endl;
    // std::cout << to_string(m) << std::endl;
    Matrix L = cholesky(m);
    std::cout << "Regular L: " << std::endl;
    std::cout << to_string(L) << std::endl;
    std::cout << "log det from chol is: " << log_det_from_cholesky(L) << std::endl;
    // std::cout << "log det directly is: " << log_det(m) << std::endl << std::endl;

    // auto row = 2;
    // std::vector<data_t> kvec;
    // auto x = {2.0, 1.0};
    // for (unsigned int i = 0; i < data.size(); ++i) {
    //     if (i == row) {
    //         kvec.push_back(1.0 + kernel(x,x) / std::pow(1.0, 2.0) - m(row,i));
    //     } else {
    //         kvec.push_back(kernel(x,data[i]) / std::pow(1.0, 2.0) - m(row,i));
    //     }
    // }

    // std::cout << "Matrix AFTER update is: " << std::endl;
    // m.rank_one_update(row, &kvec[0]);
    // std::cout << to_string(m) << std::endl;

    // // update from cholesky
    // Matrix Lnew(L);
    // // for (unsigned int k = 0; k < L.size(); ++k) {
    // //     data_t r = std::sqrt(L(k, k)*L(k, k) + kvec[k]*kvec[k]);
    // //     data_t c = r / L(k, k);
    // //     data_t s = kvec[k] / L(k, k);
    // //     L(k, k) = r;
    // //     // if (k < L.size() - 1) {
    // //     for (unsigned int i = k + 1; i < L.size(); ++i) {
    // //         L(i,k) += s*kvec[i] / c;
    // //         kvec[i] = c*kvec[i] - s*L(i,k);
    // //     }
    // //     // }
    // // }

    // data_t b = 1;
    // for (unsigned int j = 0; j < L.size(); ++j) {
    //     Lnew(j,j) = std::sqrt(L(j, j)*L(j, j) + kvec[j]*kvec[j] / b);
    //     data_t gamma = L(j,j)*b + kvec[j];
    //     for (unsigned int k = j + 1; k < L.size(); ++k) {
    //         kvec[k] -= kvec[j] / L(j,j) * L(k,j);
    //         Lnew(k,j) = Lnew(j,j) / L(j,j) * L(k,j) + Lnew(j,j) * kvec[j] / gamma * kvec[k];
    //     }
    //     b += kvec[j] * kvec[j] / (L(j,j) * L(j,j));
    // }

    // std::cout << "With Lnew: " << std::endl;
    // std::cout << to_string(L) << std::endl;
    // std::cout << "log det from chol is: " << log_det_from_cholesky(L) << std::endl;
    // std::cout << "log det directly is: " << log_det(m) << std::endl << std::endl;

    
    // std::cout << "chol after update is:\n" << to_string(Lnew) << std::endl;
    // std::cout << "logdet from chol after update is:" << log_det_from_cholesky(Lnew) << std::endl;

    // n = length(x);
    // for k = 1:n
    //     r = sqrt(L(k, k)^2 + x(k)^2);
    //     c = r / L(k, k);
    //     s = x(k) / L(k, k);
    //     L(k, k) = r;
    //     if k < n
    //         L((k+1):n, k) = (L((k+1):n, k) + s * x((k+1):n)) / c;
    //         x((k+1):n) = c * x((k+1):n) - s * L((k+1):n, k);
    //     end
    // end
    // IVM slowIVM(RBFKernel(), 1.0);
    FastIVM fastIVM(K, RBFKernel(), 1.0);
    
    Greedy greedy(K, fastIVM);
    greedy.fit(data, ids);
    std::vector<std::vector<double>> solution = greedy.get_solution();
    std::vector<idx_t> solution_ids = greedy.get_ids();
    double fval = greedy.get_fval();

    // Random random(K, [](std::vector<std::vector<data_t>> const &X){
    //     unsigned int K = X.size();
    //     data_t * kmat = new data_t[K*K];

    //     for (unsigned int i = 0; i < K; ++i) {
    //         for (unsigned int j = i; j < K; ++j) {
    //             data_t kval = kernel(X[i], X[j]);
    //             if (i == j) {
    //                 kmat[i+i*K] = 1.0 + kval / std::pow(1.0, 2.0);
    //             } else {
    //                 kmat[i+j*K] = kval / std::pow(1.0, 2.0);
    //                 kmat[j+i*K] = kval / std::pow(1.0, 2.0);
    //             }
    //         }
    //     }

    //     data_t fval = logDet(kmat, X.size(), X.size());
    //     delete [] kmat;
    //     return fval;
    // });
    // random.fit(data);
    // std::vector<std::vector<double>> solution = random.get_solution();
    // double fval = random.get_fval();

    // SieveStreaming sieve(K, [](std::vector<std::vector<data_t>> const &X){
    //     unsigned int K = X.size();
    //     data_t * kmat = new data_t[K*K];

    //     for (unsigned int i = 0; i < K; ++i) {
    //         for (unsigned int j = i; j < K; ++j) {
    //             data_t kval = kernel(X[i], X[j]);
    //             if (i == j) {
    //                 kmat[i+i*K] = 1.0 + kval / std::pow(1.0, 2.0);
    //             } else {
    //                 kmat[i+j*K] = kval / std::pow(1.0, 2.0);
    //                 kmat[j+i*K] = kval / std::pow(1.0, 2.0);
    //             }
    //         }
    //     }

    //     data_t fval = logDet(kmat, X.size(), X.size());
    //     delete [] kmat;
    //     return fval;
    // }, 2.0, 0.1);
    // sieve.fit(data);
    // std::vector<std::vector<double>> solution = sieve.get_solution();
    // double fval = sieve.get_fval();

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