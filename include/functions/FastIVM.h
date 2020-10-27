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

public:
    FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) : IVM(kernel, sigma),  K(K) {
        added = 0;
        fval = 0;
        kmat = new data_t[K*K];
        L = new data_t[K*K];
    }

    data_t peek(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {
        if (added < K) {
            for (size_t i = 0; i < added; ++i) {
                data_t kval = kernel(cur_solution[i], x);

                kmat[i * K + added] = kval / std::pow(sigma, 2.0);
                kmat[added * K + i] = kval / std::pow(sigma, 2.0);
            }
            data_t kval = kernel(x, x);
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

    void update(std::vector<std::vector<data_t>> &cur_solution, std::vector<data_t> const &x) {
        if (added < K) {
            fval = peek(cur_solution, x);
            added++;
        }
    }

    std::shared_ptr<SubmodularFunction> clone() const {
        return std::shared_ptr<SubmodularFunction>(new FastIVM(K, kernel, sigma));
    }

    ~FastIVM() {
        if (kmat != NULL) {
            delete kmat;
            kmat = NULL;
        }

        if (L != NULL) {
            delete L;
            L = NULL;
        }
    }
};

#endif // FAST_IVM_H

