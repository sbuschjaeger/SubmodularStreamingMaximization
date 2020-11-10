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
#include "functions/Matrix.h"

class IVM : public SubmodularFunction {
protected:

    inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X) const {
        unsigned int K = X.size();
        Matrix mat(K);

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel->operator()(X[i], X[j]);
                if (i == j) {
                    mat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
                } else {
                    mat(i,j) = kval / std::pow(1.0, 2.0);
                    mat(j,i) = kval / std::pow(1.0, 2.0);
                }
            }
        }

        // TODO CHECK IF THIS USES MOVE
        return mat;
    }

    std::shared_ptr<Kernel> kernel;
    data_t sigma;

public:
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
        
        Matrix kernel_mat = compute_kernel(X);
        return log_det(kernel_mat);
    } 

    std::shared_ptr<SubmodularFunction> clone() const override {
        return std::make_shared<IVM>(*kernel, sigma);
    }

    ~IVM() {}
};

#endif // INFORMATIVE_VECTOR_MACHINE_H

