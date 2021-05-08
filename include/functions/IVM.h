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

/**
 * @brief  This class implements the InformativeVectorMachine [1]:
 *  \f[
 *      f(S) = \frac{1}{2}\log\det\left(\Sigma + \sigma \cdot \mathcal I \right)
 * \f]
 *  where \f$\Sigma\f$ is the kernel matrix of all elements in the summary, \f$ \mathcal I \f$ is the \f$ K \times K \f$ identity matrix and \f$ \sigma > 0 \f$ is a scaling parameter. This implementation is lazy and slow. It recomputes \f$ \Sigma \f$ in every evaluation. For a faster and more practical alternative please have a look at the FastIVM class. This class internally uses the Matrix class for somewhat readable linear algebra.
 * 
 * 
 * 
 * [1] Herbrich, R., Lawrence, N., & Seeger, M. (2003). Fast Sparse Gaussian Process Methods: The Informative Vector Machine. In S. Becker, S. Thrun, & K. Obermayer (Eds.), Advances in Neural Information Processing Systems (Vol. 15, pp. 625–632). MIT Press. Retrieved from https://proceedings.neurips.cc/paper/2002/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf

 * @note   
 * @retval None
 */
class IVM : public SubmodularFunction {
protected:

    /**
     * @brief  Computes the kernel similarity \Sigma + \sigma \cdot \mathcal I between all pairs in X
     * @note   
     * @param  &X: The current summary 
     * @param  sigma: Scaling for main-diagonal
     * @retval The \f$K \times K\f$ kernel matrix
     */
    inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X, data_t sigma) const {
        unsigned int K = X.size();
        Matrix mat(K);

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel->operator()(X[i], X[j]);
                if (i == j) {
                    mat(i,j) = sigma * 1.0 + kval;
                } else {
                    mat(i,j) = kval;
                    mat(j,i) = kval;
                }
            }
        }

        // TODO CHECK IF THIS USES MOVE
        return mat;
    }

    // The kernel
    std::shared_ptr<Kernel> kernel;

    // The scaling constant
    data_t sigma;

public:

    /**
     * @brief  Creates a new IVM object with the given parameters.
     * @note   This constructor uses assert to make sure that sigma has the correct range. This may lead to warnings during compilation.
     * @param  &kernel: The kernel to be used.
     * @param  sigma: The scaling constant 
     * @retval A new IVM object 
     */
    IVM(Kernel const &kernel, data_t sigma) : kernel(kernel.clone()), sigma(sigma) {
        assert(("The sigma value of the IVM should be greater than  0!", sigma > 0));
    }

    /**
     * @brief  Creates a new IVM object with the given parameters.
     * @note   This constructor uses assert to make sure that sigma has the correct range. This may lead to warnings during compilation.
     * @param  kernel: The kernel to be used.
     * @param  sigma: The scaling constant 
     * @retval A new IVM object 
     */
    IVM(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : kernel(std::unique_ptr<Kernel>(new KernelWrapper(kernel))), sigma(sigma) {
        assert(("The sigma value of the IVM should be greater than  0!", sigma > 0));
    }

    /**
     * @brief  Peek operator for the IVM. For more details see SubmodularFunction. This implementation simply adds / replaces the current item in the summary and recomputes the kernel matrix as well as the log-determinant. The runtime of this implementation is O(K^3) with K = cur_solution.size()
     * @note   The log-determinant is computed via a cholesky decomposition.
     * @param  cur_solution: The current summary
     * @param  &x: The element which should be added to the summary
     * @param  pos: The position at which the given element should be inserted. If pos >= cur_solution.size(), then x is appended. Otherwise it replaces the element at position pos
     * @retval The log-determinant of the kernel matrix, if x would be inserted at position pos in current_solution
     */
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

    /**
     * @brief  Does nothing and only exists for compatibility reasons.
     * @note   
     * @param  &cur_solution: 
     * @param  &x: 
     * @param  pos: 
     * @retval None
     */
    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {}

    /**
     * @brief  Computes the kernel matrix \Sigma + \sigma \cdot \mathcal I between all pairs in X and its log-determinant.  The runtime is O(K^3) where K = X.size().
     * @note   The log-determinant is computed via a cholesky decomposition.
     * @param  &X: The argument at which \f$f(X) = \frac{1}{2}\log\det\left(\Sigma + \sigma \cdot \mathcal I \right)\f$ should be evaluated
     * @retval The log-determinant of the kernel matrix of all pairs in X
     */
    data_t operator()(std::vector<std::vector<data_t>> const &X) const override {
        // This is the most basic implementations which recomputes everything with each call
        // I would not use this for any real-world problems. 
        
        Matrix kernel_mat = compute_kernel(X);
        return log_det(kernel_mat);
    } 

    /**
     * @brief  Clones the current IVM object.
     * @note   Calls the clone method of the given kernel. If the kernel implements a deep-copy, then this clone operation is also a deep -opy. Otherwise it is not.
     * @retval The cloned object.
     */
    std::shared_ptr<SubmodularFunction> clone() const override {
        return std::make_shared<IVM>(*kernel, sigma);
    }

    /**
     * @brief  Destroys the current object.
     * @note   
     * @retval 
     */
    ~IVM() {/* Nothing do to, since the shared_pointer should clean-up itself*/ }
};

#endif // INFORMATIVE_VECTOR_MACHINE_H

