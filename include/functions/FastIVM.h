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
#include "functions/IVM.h"

/**
 * @brief  This is a faster implementation of the IVM [1]
 * \f[
 *      f(S) = \frac{1}{2}\log\det\left(\Sigma + \sigma \cdot \mathcal I \right)
 * \f]
 *  where \Sigma is the kernel matrix of all elements in the summary, \mathcal I is the K \times K identity matrix and \sigma > 0 is a scaling parameter. 
 * 
 * This implementation caches the current kernel matrix \Sigma and maintains a cholesky decomposition of it to quickly recompute the log-determinant. This implementation requires the maximum number items in the summary and the maximum size (rows and columns) of \Sigma beforehand. It allocates the appropriate memory during construction. This implementation is optimized towards adding new elements to the summary, but not replacing existing ones. Added a new row / column to a cholesky decomposition is a rank-1 update which can be performed in O(K^2) for K x K matrices. Whenever an element in the matrix must be replaced, the entire cholesky decomposition must be recomputed leading to O(K^3). This class internally uses the Matrix class for somewhat readable linear algebra. 
 * 
 * 
 * [1] Herbrich, R., Lawrence, N., & Seeger, M. (2003). Fast Sparse Gaussian Process Methods: The Informative Vector Machine. In S. Becker, S. Thrun, & K. Obermayer (Eds.), Advances in Neural Information Processing Systems (Vol. 15, pp. 625–632). MIT Press. Retrieved from https://proceedings.neurips.cc/paper/2002/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf 
 * @note   
 * @retval None
 */
class FastIVM : public IVM {
private:
    
protected:
    // Number of items added so far. Required to maintain consistent access to kmat and L
    unsigned int added;

    // The kernel matrix \Sigma. 
    Matrix kmat;

    // The lower triangle matrix of the cholesky decomposition. Note that it stores K x K elements, even though only 1/2 * K * K + K are required for a lower triangle matrix 
    Matrix L;

    // The current function value
    data_t fval;

public:

    /**
     * @brief  Creates a new FastIVM object.
     * @note   
     * @param  K: The number of elements to be stored in the summary
     * @param  &kernel: The kernel function
     * @param  sigma: The scaling constant for the kernel
     * @retval 
     */
    FastIVM(unsigned int K, Kernel const &kernel, data_t sigma) : IVM(kernel, sigma), kmat(K+1), L(K+1) {
        added = 0;
        fval = 0;
    }

    /**
     * @brief  Creates a new FastIVM object.
     * @note   
     * @param  K: The number of elements to be stored in the summary
     * @param  &kernel: The kernel function
     * @param  sigma: The scaling constant for the kernel
     * @retval 
     */
    FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : IVM(kernel, sigma), kmat(K+1), L(K+1) {
        added = 0;
        fval = 0;
    }

    /**
     * @brief  Peek operator for the FastIVM. For more details see SubmodularFunction. This function adds the vector of kernel evaluations to the current kernel matrix and performs a rank-1 update to the cholesky decomposition, if possible. When a new element is added (pos >= added) then the runtime is O(K^2) where K = cur_solution.size() and added is the number of previous `update` calls. If an existing element is replaced (pos < added), then the cholesky decomposition cannot be updated with a rank-1 update. In this case the runtime is O(K^3) since the cholesky decomposition is recomputed. 
     * 
     * @note  
     * @param  cur_solution: The current summary
     * @param  &x: The element which should be added to the summary
     * @param  pos: The position at which the given element should be inserted. If pos >= cur_solution.size(), then x is appended. Otherwise it replaces the element at position pos
     * @retval The log-determinant of the kernel matrix, if x would be inserted at position pos in current_solution
     */
    data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos >= added) {
            // Peek function value for last line

            for (unsigned int i = 0; i < added; ++i) {
                data_t kval = kernel->operator()(cur_solution[i], x);

                kmat(i, added) = kval;
                kmat(added, i) = kval;
            }
            data_t kval = kernel->operator()(x, x);
            kmat(added, added) = sigma * 1.0 + kval;

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
                    tmp(pos, pos) = sigma * 1.0 + kval;
                } else {
                    data_t kval = kernel->operator()(cur_solution[i], x);
                    tmp(i, pos) = kval;
                    tmp(pos, i) = kval;
                }
            }

            return log_det(tmp);
        }
    }

    /**
     * @brief  Update the current solution. Does the same as `peek` and additionally preserves any changes to the kernel matrix.  
     * @note  
     * @param  cur_solution: The current summary
     * @param  &x: The element which should be added to the summary
     * @param  pos: The position at which the given element should be inserted. If pos >= cur_solution.size(), then x is appended. Otherwise it replaces the element at position pos
     */
    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        if (pos >= added) {
            // TODO We often have the peek () -> update() pattern. This call can be optimized since we now basically peek twice
            fval = peek(cur_solution, x, pos);
            added++;
        } else {
            for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                if (i == pos) {
                    data_t kval = kernel->operator()(x, x);
                    kmat(pos, pos) = sigma * 1.0 + kval;
                } else {
                    data_t kval = kernel->operator()(cur_solution[i], x);
                    kmat(i, pos) = kval;
                    kmat(pos, i) = kval;
                }
            }
            L = cholesky(kmat, added);
            fval = log_det_from_cholesky(L);
        }

    }

    /**
     * @brief  Returns the current function value which has been computed and cached during the `update` calls. The function value _does not_ depend on cur_solution in this case, but only on the order and values supplied during `update` calls to this object.
     * @note   The runtime is O(1). Nothing is computed.
     * @param  &cur_solution: Has no effect
     * @retval The log-determinant of the kernel matrix supplied during `update`
     */
    data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
        return fval;
    }

    /**
     * @brief  Clones the current object. The cloned object has an empty kernel matrix. No values are copied.
     * @note   Calls the clone method of the given kernel. If the kernel implements a deep-copy, then the cloned kernel is a deep copy. Besides that, this is _not_ a deep copy. 
     * @retval The cloned object.
     */
    std::shared_ptr<SubmodularFunction> clone() const override {
        // We want to store k elements. To allow for efficient peeking we will reserve space for K + 1 elements in kmat and L. 
        // Thus we need to call the constructor with one element less
        return std::make_shared<FastIVM>(kmat.size() - 1, *kernel, sigma);
    }
};

#endif // FAST_IVM_H

