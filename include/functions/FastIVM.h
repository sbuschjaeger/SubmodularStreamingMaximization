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
    unsigned int added;
    Matrix kmat;
    Matrix L;
    data_t fval;

public:
    FastIVM(unsigned int K, Kernel const &kernel, data_t sigma) : IVM(kernel, sigma), kmat(K+1), L(K+1) {
        added = 0;
        fval = 0;
    }

    FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : IVM(kernel, sigma), kmat(K+1), L(K+1) {
        added = 0;
        fval = 0;
    }

    data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos >= added) {
            // Peek function value for last line

            for (unsigned int i = 0; i < added; ++i) {
                data_t kval = kernel->operator()(cur_solution[i], x);

                kmat(i, added) = kval / std::pow(sigma, 2.0);
                kmat(added, i) = kval / std::pow(sigma, 2.0);
            }
            data_t kval = kernel->operator()(x, x);
            kmat(added, added) = 1.0 + kval / std::pow(sigma, 2.0);

            for (size_t j = 0; j <= added; j++) {
                //data_t s = std::inner_product(&L[added * K], &L[added * K] + j, &L[j * K], static_cast<data_t>(0));
                data_t s = std::inner_product(&L(added, 0), &L(added, j), &L(j,0), static_cast<data_t>(0));
                if (added == j) {
                    L(added, j) = std::sqrt(kmat(added, j) - s);
                } else {
                    L(added, j) = (1.0f / L(j, j) * (kmat(added, j) - s));
                }
                L(j, added) = L(added, j); // Symmetric update
            }
            return fval + 2.0 * std::log(L(added, added));
        } else {
            Matrix tmp(kmat, added);
            for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                if (i == pos) {
                    data_t kval = kernel->operator()(x, x);
                    tmp(pos, pos) = 1.0 + kval / std::pow(sigma, 2.0);
                } else {
                    data_t kval = kernel->operator()(cur_solution[i], x);
                    tmp(i, pos) = kval / std::pow(sigma, 2.0);
                    tmp(pos, i) = kval / std::pow(sigma, 2.0);
                }
            }

            return log_det(tmp);
        }
    }
        // if (pos < cur_solution.size()) {
        //     std::vector<std::vector<data_t>> tmp(cur_solution);
        //     tmp[pos] = x;
        //     return IVM::operator()(tmp);
        // } else {
        //     //if (added < K) {
        //     for (size_t i = 0; i < added; ++i) {
        //         data_t kval = kernel->operator()(cur_solution[i], x);

        //         kmat[i * K + added] = kval / std::pow(sigma, 2.0);
        //         kmat[added * K + i] = kval / std::pow(sigma, 2.0);
        //     }
        //     data_t kval = kernel->operator()(x, x);
        //     kmat[added+added*K] = 1.0 + kval / std::pow(sigma, 2.0);

        //     for (size_t j = 0; j <= added; j++) {
        //         data_t s = std::inner_product(&L[added * K], &L[added * K] + j, &L[j * K], static_cast<data_t>(0));
        //         if (added == j) {
        //             L[added * K + j] = std::sqrt(kmat[added * K + j] - s);
        //         } else {
        //             L[added * K + j] = (1.0f / L[j * K + j] * (kmat[added * K + j] - s));
        //         }
        //         L[j * K + added] = L[added * K + j]; // Symmetric update
        //     }
        //     return fval + 2.0 * std::log(L[added*K + added]);
            // } else {
            //     return 0;
            // }
    //     }
    // }

    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos >= added) {
            // We often have the peek () -> update() pattern. This call can be optimized since we now basically peek twice
            fval = peek(cur_solution, x, pos);
            added++;
        } else {
            for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                if (i == pos) {
                    data_t kval = kernel->operator()(x, x);
                    kmat(pos, pos) = 1.0 + kval / std::pow(sigma, 2.0);
                } else {
                    data_t kval = kernel->operator()(cur_solution[i], x);
                    kmat(i, pos) = kval / std::pow(sigma, 2.0);
                    kmat(pos, i) = kval / std::pow(sigma, 2.0);
                }
            }
            L = cholesky(kmat, added);
            fval = log_det_from_cholesky(L);
        }

        // fval = log_det_from_cholesky(L);
        // if (pos < cur_solution.size()) {
        //     // TODO CHANGE THIS
        //     std::vector<std::vector<data_t>> tmp_solution(cur_solution);
        //     tmp_solution[pos] = x;
        //     unsigned int K = tmp_solution.size();
        //     IVM::compute_kernel(kmat, tmp_solution);
        //     data_t * tmp = new data_t[K*K];
        //     for (size_t i = 0; i < K; ++i) {
        //         for (size_t j = 0; j < K; ++j) {
        //             tmp[i*K+j] = kmat[i*K+j];
        //         }
        //     }
        //     IVM::cholesky(L, tmp, K, K);
        //     data_t det = 0;
        //     for (unsigned int i = 0; i < K; ++i) {
        //         det += std::log(L[i * K + i]);
        //     }
        //     added = cur_solution.size();
        //     fval = 2*det;
        //     delete[] tmp;
        // } else {
        //     if (cur_solution.size() < K) {
        //         fval = peek(cur_solution, x, pos);
        //         added = cur_solution.size();
        //     }
        // }
    }

    data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
        return fval;
    }

    std::shared_ptr<SubmodularFunction> clone() const override {
        // We want to store k elements. To allow for efficient peeking we will reserve space for K + 1 elements in kmat and L. 
        // Thus we need to call the constructor with one element less
        return std::make_shared<FastIVM>(kmat.size() - 1, *kernel, sigma);
    }
};

#endif // FAST_IVM_H

