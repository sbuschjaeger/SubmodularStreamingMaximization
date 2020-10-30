#ifndef FAST_IVM_H
#define FAST_IVM_H

#include <mutex>
#include <vector>
#include <functional>
#include <math.h>
#include <cassert>
#include <numeric>

#include "DataTypeHandling.h"
#include "SubmodularFunction.h"
#include "IVM.h"

//#include <src/SubmodularFunction.h>
//#include "kernels/Kernel.h"
/**
 *
 */
class FastIVM : public IVM {
private:
    
protected:
    unsigned int K;
    unsigned int added;
    data_t * kmat;
    data_t * L;
    data_t fval;

    inline static data_t _kernel(const std::vector<data_t>& x1, const std::vector<data_t>& x2) {
        data_t distance = 0;
        if (x1 != x2) {
            for (unsigned int i = 0; i < x1.size(); ++i) {
                distance += (x1[i]-x2[i])*(x1[i]-x2[i]);
            }
            distance /= std::sqrt(x1.size());
        }
        return 1.0 * std::exp(-distance);
    }

public:
    //FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> const &kernel, data_t sigma) : IVM(kernel, sigma),  K(K) {
    FastIVM(unsigned int K, Kernel const &kernel, data_t sigma) : IVM(kernel, sigma),  K(K) {
        added = 0;
        fval = 0;
        kmat = new data_t[K*K];
        L = new data_t[K*K];
    }

    FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : IVM(kernel, sigma),  K(K) {
        added = 0;
        fval = 0;
        kmat = new data_t[K*K];
        L = new data_t[K*K];
    }

    data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos < cur_solution.size()) {
            std::vector<std::vector<data_t>> tmp(cur_solution);
            tmp[pos] = x;
            return IVM::operator()(tmp);
        } else {
            if (added < K) {
                for (size_t i = 0; i < added; ++i) {
                    data_t kval = kernel->operator()(cur_solution[i], x);

                    kmat[i * K + added] = kval / std::pow(sigma, 2.0);
                    kmat[added * K + i] = kval / std::pow(sigma, 2.0);
                }
                data_t kval = kernel->operator()(x, x);
                kmat[added+added*K] = 1.0 + kval / std::pow(sigma, 2.0);

                for (size_t j = 0; j <= added; j++) {
                    data_t s = std::inner_product(&L[added * K], &L[added * K] + j, &L[j * K], static_cast<data_t>(0));
                    if (added == j) {
                        L[added * K + j] = std::sqrt(kmat[added * K + j] - s);
                    } else {
                        L[added * K + j] = (1.0f / L[j * K + j] * (kmat[added * K + j] - s));
                    }
                    L[j * K + added] = L[added * K + j]; // Symmetric update
                }
                return fval + 2.0 * std::log(L[added*K + added]);
            } else {
                return 0;
            }
        }
    }

    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos < cur_solution.size()) {
            // TODO CHANGE THIS
            std::vector<std::vector<data_t>> tmp_solution(cur_solution);
            tmp_solution[pos] = x;
            unsigned int K = tmp_solution.size();
            IVM::compute_kernel(kmat, tmp_solution);

            data_t * tmp = new data_t[K*K];
            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    tmp[i*K+j] = kmat[i*K+j];
                }
            }
            IVM::cholesky(L, tmp, K, K);
            
            data_t det = 0;
            for (unsigned int i = 0; i < K; ++i) {
                det += std::log(L[i * K + i]);
            }

            added = K;
            fval = 2*det;
            delete[] tmp;
        } else {
            if (added < K) {
                fval = peek(cur_solution, x, pos);
                added++;
            }
        }
    }

    data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
        return fval;
    }

    std::shared_ptr<SubmodularFunction> clone() const override {
        return std::make_shared<FastIVM>(K, *kernel, sigma);
    }

    ~FastIVM() {
        if (kmat != NULL) {
            delete[] kmat;
            kmat = NULL;
        }

        if (L != NULL) {
            delete[] L;
            L = NULL;
        }
    }
};

#endif // FAST_IVM_H

