#ifndef RANDOM_H
#define RANDOM_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>


/**
 * @brief The Random Optimizers for submodular functions. It randomly picks K elements as a solution. For streaming, Reservoir Sampling is used. Feige et al. showed in [1] that a uniform random sample for unconstrained is a 1/4 approximation in expectation, but for constraint maximization problems no such result is known.
 *  - Stream:  Yes
 *  - Solution: no guarantee
 *  - Runtime: \f$ O(N) \f$
 *  - Memory: \f$ O(K) \f$
 *  - Function Queries per Element: \f$ O(1) \f$ (In case of Reservoir Sampling to maintain a consistent function value)
 *  - Function Types: nonnegative submodular functions
 * 
 * __References__:
 * 
 * [1] Feige, U., Mirrokni, V. S., & Vondrák, J. (2011). Maximizing non-monotone submodular functions. SIAM Journal on Computing. https://doi.org/10.1137/090779346
 * [2] Vitter, J. S. (1985). Random Sampling with a Reservoir. ACM Transactions on Mathematical Software (TOMS). https://doi.org/10.1145/3147.3165
 * @note   
 */
class Random : public SubmodularOptimizer {
protected:
    unsigned int cnt = 0;
    std::default_random_engine generator;

    /**
     * @brief  Sample k elements from the range \f$ [1, N] \f$ without replacement. Caller needs to make sure that \f$ k \le N \f$. The runtime is \f$ O(k) \f$ and memory is \f$ O(k) \f$
     * @note   Basically copied from https://www.gormanalysis.com/blog/random-numbers-in-cpp/#sampling-without-replacement
     * @param  k: The number of samples to be selected
     * @param  N: The upper bound of the interval to sample from
     * @param  gen: A random generator used for sampling
     * @retval The sampled vector of indices
     */
    static inline std::vector<idx_t> sample_without_replacement(int k, int N, std::default_random_engine& gen) {
        // Create an unordered set to store the samples
        std::unordered_set<idx_t> samples;
        
        // Sample and insert values into samples
        for (int r = N - k; r < N; ++r) {
            idx_t v = std::uniform_int_distribution<>(1, r)(gen);
            if (!samples.insert(v).second) samples.insert(r - 1);
        }
        
        // Copy samples into vector
        std::vector<idx_t> result(samples.begin(), samples.end());
        
        // Shuffle vector
        std::shuffle(result.begin(), result.end(), gen);
        
        return result;
    };

public:

    /**
     * @brief Construct a new Random object.
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     * @param seed The random seed used for randomization.
     */
    Random(unsigned int K, SubmodularFunction & f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}

    /**
     * @brief Construct a new Random object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state. 
     * @param seed The random seed used for randomization.
     */
    Random(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}

     /**
     * @brief  Randomly pick K elements as a solution. You can access the solution via `get_solution` and the ids can be accessed via `get_ids`.
     * @note   
     * @param  X A constant reference to the entire data set
     * @param ids: A list of identifier for each object. This can be used to uniquely identify the objects in the summary. If ids.size() < X.size(), then only partial ids are stored. No ids are stored if ids is empty. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks. 
     * @param iterations: Has no effect. Random samples K elements, no iterations required.
     */
    void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
        if (X.size() < K) {
            K = X.size();
        }
        std::vector<idx_t> indices = sample_without_replacement(K, X.size(), generator);

        for (auto i : indices) {
            f->update(solution, X[i], solution.size());
            solution.push_back(X[i]);
            if (ids.size() >= i) {
                this->ids.push_back(ids[i]);
            }
            //solution.push_back(std::vector<data_t>(X[i]));
        }

        cnt = X.size();
        fval = f->operator()(solution);
        is_fitted = true;
    }

    /**
     * @brief  Randomly pick K elements as a solution. You can access the solution via `get_solution`.
     * @note   
     * @param  X A constant reference to the entire data set
     * @param iterations: Has no effect. Random samples K elements, no iterations required.
     */
    void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        std::vector<idx_t> ids;
        fit(X,ids,iterations);
    }

    /**
     * @brief Consume the next object in the data stream. This call uses Reservoir Sampling to sample the current solution which can access via `get_solution`.
     * 
     * @param x A constant reference to the next object on the stream.
     * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks. 
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        if (solution.size() < K) {
            // Just add the first K elements
            f->update(solution, x, solution.size());
            solution.push_back(x);
            if (id.has_value()) ids.push_back(id.value());
        } else {
            // Sample the replacement-index with decreasing probability
            unsigned int j = std::uniform_int_distribution<>(1, cnt)(generator);
            if (j <= K) {
                f->update(solution, x, j - 1);
                if (id.has_value()) ids[j-1] = id.value();
                solution[j - 1] = x; 
            }
        }

        // Update the current function value
        fval = f->operator()(solution);
        is_fitted = true;
        ++cnt;
    }
};

#endif // RANDOM_H
