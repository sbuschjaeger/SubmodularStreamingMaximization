#ifndef GREEDY_H
#define GREEDY_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>

/**
 * @brief  The Greedy optimizers for submodular functions. This implementation is due to Nemhauser etal.
 * @note   
 * @retval None
 */
class Greedy : public SubmodularOptimizer {
public:
    Greedy(unsigned int K, SubmodularFunction & f) : SubmodularOptimizer(K,f) {}

    Greedy(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : SubmodularOptimizer(K,f) {}

    /**
     *
     * @param dataset
     * @return
     */
    void fit(std::vector<std::vector<data_t>> const & X) {
        std::vector<unsigned int> remaining(X.size());
        std::iota(remaining.begin(), remaining.end(), 0);
        data_t fcur = 0;

        while(solution.size() < K && remaining.size() > 0) {
            std::vector<data_t> fvals;
            fvals.reserve(remaining.size());
            
            // Technically the Greedy algorithms picks that element with largest gain. This is equivalent to picking that
            // element which results in the largest function value. There is no need to explicitly compute the gain
            for (auto i : remaining) {
                data_t ftmp = f->peek(solution, X[i], solution.size());
                fvals.push_back(ftmp);
            }

            unsigned int max_element = std::distance(fvals.begin(),std::max_element(fvals.begin(), fvals.end()));
            fcur = fvals[max_element];
            unsigned int max_idx = remaining[max_element];
            
            // Copy new vector into solution vector
            f->update(solution, X[max_idx], solution.size());
            //solution.push_back(std::vector<data_t>(X[max_idx]));
            solution.push_back(X[max_idx]);
            remaining.erase(remaining.begin()+max_element);
        }

        fval = fcur;
        is_fitted = true;
    }

    /**
     *
     * @param dataset
     * @return
     */
    void next(std::vector<data_t> const &x) {
        throw std::runtime_error("Greedy does not support streaming data, please use fit().");
    }

    
};

#endif // GREEDY_H
