#ifndef GREEDY_H
#define GREEDY_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>

/**
 * @brief  The Greedy optimizer for submodular functions. It rates the marginal gain of each element and picks that element with the largest gain. This process is repeated until it K elements have been selected:
 *  - Stream:  No
 *  - Solution: \f$ 1 - 1/\exp(1) \f$
 *  - Runtime: \f$ O(N \cdot K) \f$
 *  - Memory: \f$ O(K) \f$
 *  - Function Queries per Element: \f$ O(1) \f$
 *  - Function Types: nonnegative submodular functions
 * 
 * Example usage in C++:
 * @code{.cpp}
 *  //read some data 
 *  std::vector<std::vector<data_t>> = read_some_data(); 
 *  auto K = 50;
 *  // Define the function to be maximized and select the summary
 *  FastIVM fastIVM(K, RBFKernel( std::sqrt(data[0].size()), 1.0) , 1.0);
 *  Greedy opt(K, fastIVM);
 *  opt.fit(data);
 *  std::cout << "fval:" << opt.get_fval() << "num_elements: " << opt.get_num_elements_stored() << "num_candidates: " << opt.get_num_candidate_solutions() << std::endl;
 *  // Process summary
 *  auto summary = opt.get_solution();
 * @endcode
 * 
 * Example usage in Python:
 * @code{.py}
 *  X = read_some_data(); 
 *  K = 50
 *  # Create function to be maximized
 *  kernel = RBFKernel(sigma=sigma,scale=scale)
 *  fastLogDet = FastIVM(K, kernel, 1.0)
 *  opt = Greedy(K, fastLogDet)
 *  opt.fit(X, K)
 *  print("fval: {} num_elements: {} num_candidates: {}".format(opt.get_fval(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions()))
 *  # process summary
 *  summary = opt.get_solution()
 * @endcode
 * 
 * __References__
 * 
 * - Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions-I. Mathematical Programming, 14(1), 265–294. https://doi.org/10.1007/BF01588971
 */
class Greedy : public SubmodularOptimizer {
public:
    
    /**
     * @brief Construct a new Greedy object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     */
    Greedy(unsigned int K, SubmodularFunction & f) : SubmodularOptimizer(K,f) {}


    /**
     * @brief Construct a new Greedy object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if `f` keeps track of a state.
     */
    Greedy(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : SubmodularOptimizer(K,f) {}

    /**
     * @brief Pick that element with the largest marginal gain in the entire dataset. Repeat this until K element have been selected. You can access the solution via `get_solution` and the corresponding ids (if passed) with `get_ids()`
     * 
     * @param X A constant reference to the entire data set
     * @param ids: A list of identifier for each object. This can be used to uniquely identify the objects in the summary. If ids.size() < X.size(), then only partial ids are stored. No ids are stored if ids is empty. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks. 
     * @param iterations: Has no effect. Greedy iterates K times over the entire dataset in any case.
     */
    void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
    //void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
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
            if (ids.size() >= max_idx) {
                this->ids.push_back(max_idx);
            }
            remaining.erase(remaining.begin()+max_element);
        }

        fval = fcur;
        is_fitted = true;
    }

    /**
     * @brief Pick that element with the largest marginal gain in the entire dataset. Repeat this until K element have been selected. You can access the solution via `get_solution`
     * @note This internally calls fit with an empty id set.
     * @param X A constant reference to the entire data set
     * @param iterations: Has no effect. Greedy iterates K times over the entire dataset in any case.
     */
    void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        std::vector<idx_t> ids;
        fit(X,ids,iterations);
    }


    /**
     * @brief Throws an exception when called. Greedy does not support streaming!
     * 
     * @param x A constant reference to the next object on the stream.
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> id = std::nullopt) {
        throw std::runtime_error("Greedy does not support streaming data, please use fit().");
    }

    
};

#endif // GREEDY_H
