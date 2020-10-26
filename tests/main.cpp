#include <iostream>
#include <vector>
#include <math.h>

#include "functions/FastIVM.h"
#include "kernels/RBFKernel.h"
#include "Greedy.h"


void cholesky(data_t *const pOut, data_t const *const pIn, size_t const N, size_t const ld) {
    for (unsigned int j = 0; j < N; ++j) {
        data_t sum = 0.0;

        for (unsigned int k = 0; k < j; ++k) {
            sum += pOut[j * N + k] * pOut[j * N + k];
        }

        pOut[j * N + j] = std::sqrt(pIn[j * ld + j] - sum);

        for (unsigned int i = j + 1; i < N; ++i) {
            data_t sum = 0.0;

            for (unsigned int k = 0; k < j; ++k) {
                sum += pOut[i * N + k] * pOut[j * N + k];
            }

            pOut[i * N + j] = (pIn[i * ld + j] - sum) / pOut[j * N + j];
        }
    }
}

data_t logDet(data_t const *const pM, size_t const N, size_t ld) {
    data_t *L = new data_t[N * N];

    data_t * tmp = new data_t[N*N];
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            tmp[i*N+j] = pM[i*ld+j];
        }
    }

    cholesky(L, tmp, N, N);

    //std::cout << "cholesky: " << toNumpy(L, N, N) << std :: endl;

    data_t det = 0;
    // TODO: THIS CAN BE IMPROVED / MAYBE THERE IS A MKL/LAPACK FUNCTION FOR THIS
    for (size_t i = 0; i < N; ++i) {
        //if (L[i * N + i] == 0) det += std::numeric_limits<T>::max();
        det += std::log(L[i * N + i]);
    }

    delete[] tmp;

    delete[] L;
    return 2*det;
}

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

int main() {

    std::vector<std::vector<double>> data = {
        {0, 0},
        {1, 1},
        {0.5, 1.0},
        {1.0, 0.5},
        {0, 0.5},
        {0.5, 1},
        {0.0, 1.0},
        {1.0, 0.0}
    };    

    unsigned int K = 3;
    // IVM slowIVM(RBFKernel(), 1.0);
    // Greedy greedy(3, &slowIVM);
    //auto fastIVM = new FastIVM(K, RBFKernel(), 1.0);
    // FastIVM * fastIVM = new FastIVM(K, RBFKernel(), 1.0);

    FastIVM fastIVM(K, RBFKernel(), 1.0);
    IVM slowIVM(RBFKernel(), 1.0);
    Greedy greedy(K, [](std::vector<std::vector<data_t>> const &X){
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
    });

    greedy.fit(data);
    std::vector<std::vector<double>> solution = greedy.get_solution();
    double fval = greedy.get_fval();

    std::cout << "Found a solution with fval = " << fval << std::endl;
    for (auto x : solution) {
        for (auto xi : x) {
            std::cout << xi << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}