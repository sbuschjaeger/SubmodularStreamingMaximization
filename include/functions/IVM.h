#ifndef INFORMATIVE_VECTOR_MACHINE_H
#define INFORMATIVE_VECTOR_MACHINE_H

#include <mutex>
#include <vector>
#include <functional>
#include <math.h>
#include <cassert>
#include "DataTypeHandling.h"
#include "SubmodularFunction.h"
#include "kernels/Kernel.h"

//#include <src/SubmodularFunction.h>
//#include "kernels/Kernel.h"
/**
 *
 */
class IVM : public SubmodularFunction {
protected:

    inline void compute_kernel(data_t *const kmat, std::vector<std::vector<data_t>> const &X) const {
        unsigned int K = X.size();

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel->operator()(X[i], X[j]);
                if (i == j) {
                    kmat[i+i*K] = 1.0 + kval / std::pow(sigma, 2.0);
                } else {
                    kmat[i+j*K] = kval / std::pow(sigma, 2.0);
                    kmat[j+i*K] = kval / std::pow(sigma, 2.0);
                }
            }
        }
    }

    static inline void cholesky(data_t *const pOut, data_t const *const pIn, size_t const N, size_t const ld) {
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

    static inline data_t log_det(data_t const *const pM, size_t const N, size_t ld) {
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

    //std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel;
    std::shared_ptr<Kernel> kernel;
    data_t sigma;

public:
    //IVM(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> const &kernel, data_t sigma) : kernel(kernel), sigma(sigma) {}
    IVM(Kernel const &kernel, data_t sigma) : kernel(kernel.clone()), sigma(sigma) {}

    IVM(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : kernel(std::unique_ptr<Kernel>(new KernelWrapper(kernel))), sigma(sigma) {
    }

    data_t peek(std::vector<std::vector<data_t>> const& cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        std::vector<std::vector<data_t>> tmp(cur_solution);

        if (pos >= cur_solution.size()) {
            tmp.push_back(x);
        } else {
            tmp[pos] = x;
        }

        data_t ftmp = this->operator()(tmp);
        return ftmp;
    } 

    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {}

    data_t operator()(std::vector<std::vector<data_t>> const &X) const override {
        // This is the most basic implementations which recomputes everything with each call
        // I would not use this for any real-world problems. 

        unsigned int K = X.size();
        data_t * kmat = new data_t[K*K];
        compute_kernel(kmat, X);

        data_t fval = IVM::log_det(kmat, X.size(), X.size());
        delete [] kmat;
        return fval;
    }

    std::shared_ptr<SubmodularFunction> clone() const override {
        return std::shared_ptr<SubmodularFunction>(new IVM(*kernel, sigma));
    }

    ~IVM() {}
};

#endif // INFORMATIVE_VECTOR_MACHINE_H

