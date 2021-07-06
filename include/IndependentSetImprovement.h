#ifndef INDEPENDENT_SET_IMPROVEMENT_H
#define INDEPENDENT_SET_IMPROVEMENT_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <string>
#include <queue>

#include <iostream>

/**
 * @brief  Independent Set Improvement for submodular functions. This optimizer stores the marginal gain ("weight") of each element upon arrival and replaces an element if its gain is at-least twice as large as the smallest gain currently stored in the summary. The gains are __not__ recomputed if the summary changes and thus are somewhat independent from the current solution and hence the name: 
 *  - Stream:  No
 *  - Solution: \f$ 1/4 \f$
 *  - Runtime: \f$ O(N) \f$
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
 *  IndependentSetImprovement opt(K, fastIVM);
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
 *  opt = IndependentSetImprovement(K, fastLogDet)
 *  opt.fit(X, K)
 *  print("fval: {} num_elements: {} num_candidates: {}".format(opt.get_fval(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions()))
 *  # process summary
 *  summary = opt.get_solution()
 * @endcode
 * 
 * Parts of this algorithms are proposed in the appendix of the following paper, but this might not be enough to fully replicate the algorithm. Many thanks to Sagar Kale who helped with the implementation by answering my many questions via email.
 * 
 * __References__
 * 
 * - Chakrabarti, Amit, and Sagar Kale. "Submodular maximization meets streaming: Matchings, matroids, and more." Mathematical Programming 154.1 (2015): 225-247.
 * 
 * @note   This implementation uses a priority queue for managing the weights of each item. Thus, there is a \f$ O(log K) \f$ overhead when inserting new elements. 
 */
class IndependentSetImprovement : public SubmodularOptimizer {

protected:

    /**
     * @brief  We use a priority queue to efficiently find / manage the smallest weights. Each item is identified by its weight and index in the summary. 
     */
    struct Pair {
        // The weight
        data_t weight;

        // The index in the summary
        unsigned int idx;

        /**
         * @brief  Creates a new Pair object with the given weight and index.
         * @param  _weight: The weight of the element
         * @param  _idx: The position / index in the summary
         */
        Pair(data_t _weight, unsigned int _idx) {
            weight = _weight;
            idx = _idx;
        }

        /**
         * @brief  Implementation of the comparison operator for sorting into the priority queue. Sorting is done on the basis of the objects weight:
         *          this->weight > other.weight
         * @param  &other: The other object this object is compared against.
         */
        bool operator < (const Pair &other) const { 
            return weight > other.weight; 
        } 
    };

    // The priority queue
    std::priority_queue<Pair> weights; 
public:

    /**
     * @brief Construct a new IndependentSetImprovement object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     */
    IndependentSetImprovement(unsigned int K, SubmodularFunction & f) : SubmodularOptimizer(K,f)  {
    }   

    /**
     * @brief Construct a new IndependentSetImprovement object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     */
    IndependentSetImprovement(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : SubmodularOptimizer(K,f) {
    }
    

    /**
     * @brief  Consume the next object in the data stream:
     * If there are fewer than K elements in the summary: Unconditionally accept the current, compute the function value and weight and update the priority queue of weights. Runtime is \f$ O(log K) + 1 \f$ function query 
     * If there are more than K elements in the summary: Compute the current function value and check if the weight is at-least twice as large as the smallest weight in the summary. If so, replace it. Runtime is \f$ O(1) \f$ (no update) or \f$ O(log K) \f$ (insert new element) + 1 function query
     * 
     * @param  &x: A constant reference to the next object on the stream.
     * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
     * @retval None
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        unsigned int Kcur = solution.size();
        
        if (Kcur < K) {
            data_t w = f->peek(solution, x, solution.size()) - fval;
            f->update(solution, x, solution.size());
            solution.push_back(x);
            if (id.has_value()) ids.push_back(id.value());
            weights.push(Pair(w, Kcur));
        } else {
            Pair to_replace = weights.top();
            data_t w = f->peek(solution, x, solution.size()) - fval;
            if (w > 2*to_replace.weight) {
                f->update(solution, x, to_replace.idx);
                solution[to_replace.idx] = x; 
                if (id.has_value()) ids[to_replace.idx] = id.value();
                weights.pop();
                weights.push(Pair(w, to_replace.idx));
            }
        }
        fval = f->operator()(solution);
        is_fitted = true;
    }
};

#endif