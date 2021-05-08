#ifndef SIEVESTREAMING_H
#define SIEVESTREAMING_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

/**
 * @brief Samples a set of thresholds from \f$ {(1+epsilon)^i  | i \in Z, lower \le (1+epsilon)^i \le upper} \f$ as described in [1]
 * 
 * __References :__
 * [1] Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., & Krause, A. (2014). Streaming submodular maximization: Massive data summarization on the fly. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2623330.2623637
 * @param lower The lower bound (inclusive) which is used form sampling
 * @param upper The upper bound (inclusive) which is used form sampling
 * @param epsilon The sampling accuracy
 * @return std::vector<data_t> The set of sampled thresholds
 */
inline std::vector<data_t> thresholds(data_t lower, data_t upper, data_t epsilon) {
    std::vector<data_t> ts;

    if (epsilon > 0.0) {
        // int i = std::ceil(std::log(lower) / std::log(1.0 + epsilon));
        // data_t val = std::pow(1+epsilon, i);

        // while( val < upper) {
        //     val = std::pow(1+epsilon, i);
        //     ts.push_back(val);
        //     ++i;
        // }

        int ilower = std::ceil(std::log(lower) / std::log(1.0 + epsilon));
        //int iupper; // = std::floor(std::log(upper) / std::log(1.0 + epsilon));
        // data_t tmp = std::log(upper) / std::log(1.0 + epsilon);
        // if (tmp == std::floor(tmp)) {
        //     iupper = std::floor(tmp) - 1;
        // } else {
        //     iupper = std::floor(tmp);
        // }

        // if (ilower >= upper)
        //     throw std::runtime_error("thresholds: Lower threshold boundary (" + std::to_string(ilower) + ") is higher than or equal to the upper boundary ("
        //                             + std::to_string(upper) + "), epsilon = " + std::to_string(epsilon) + ".");

        for (data_t val = std::pow(1.0 + epsilon, ilower); val <= upper; ++ilower, val = std::pow(1.0 + epsilon, ilower)) {
            ts.push_back(val);
        }
    } else {
        throw std::runtime_error("thresholds: epsilon must be a positive real-number (is: " + std::to_string(epsilon) + ").");
    }
    
    return ts;
}

/** 
 * @brief The SieveStreaming optimizer for nonnegative, monotone submodular functions. It tries to estimate the potential gain of an element ahead of time by sampling different thresholds from \f$ {(1+epsilon)^i  | i \in Z, lower \le (1+epsilon)^i \le upper} \f$ and maintaining a set of sieves in parallel. Each sieve uses a different threshold to sieve-out elements with too few of a gain. 
 *  - lower = \f$ max_e f({e}) \f$  which is the largest function value of a singleton-set
 *  - upper = \f$ K \cdot max_e f({e}) \f$  which is \f$ K \f$ times the function value of a singleton-set
 
 *  - Stream:  Yes
 *  - Solution: \f$ 1/2 - \varepsilon \f$
 *  - Runtime: \f$ O(1) \f$
 *  - Memory: \f$ O(K \cdot log(K) / \varepsilon) \f$
 *  - Function Queries per Element: \f$ O(log(K) / \varepsilon) \f$
 *  - Function Types: nonnegative, monotone submodular functions
 * 
 * __References__
 * 
 * [1] Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., & Krause, A. (2014). Streaming submodular maximization: Massive data summarization on the fly. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2623330.2623637
 */
class SieveStreaming : public SubmodularOptimizer {
private:

    /**
     * @brief  A single sieve with its own threshold and accompanying summary.  
     * @note   This class is basically also implemented in SieveStreamingPP and - to some extend - in Salsa. I decided against a unified class for these Sieves, since the thresholding rules are often slightly different from paper to paper. I tried to stick as close as possible to the pseudocode in the papers.
     * @retval None
     */
    class Sieve : public SubmodularOptimizer {
    public:
        // The threshold
        data_t threshold;

        /**
         * @brief Construct a new Sieve object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
         * @param threshold The threshold.
         */
        Sieve(unsigned int K, SubmodularFunction & f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        /**
         * @brief Construct a new Sieve object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
         * @param threshold The threshold.
         */
        Sieve(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {
        }

        /**
         * @brief Throws an exception since fit() should not be used directly here. Sieves are not meant to be used on their own, but only through SieveStreaming.
         * 
         * @param X A constant reference to the entire data set
         */
        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
        }

        /**
         * @brief Consume the next object in the data stream. This call compares the marginal gain against the given threshold and add the current item to the current solution if it exceeds the given threshold. 
         * 
         * @note   
         * @param  &x: A constant reference to the next object on the stream.
         * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
         * @retval None
         */
        void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;
                data_t tau = (threshold / 2.0 - fval) / static_cast<data_t>(K - Kcur);

                if (fdelta >= tau) {
                    f->update(solution, x, solution.size());
                    solution.push_back(x);
                    if (id.has_value()) ids.push_back(id.value());
                    fval += fdelta;
                }
            }
            is_fitted = true;
        }
    };

protected:
    // A list of all sieves
    std::vector<std::unique_ptr<Sieve>> sieves;

public:

    /**
     * @brief Construct a new SieveStreaming object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     * @param m The maximum value of the singleton set, m = max_e f({e}) 
     * @param epsilon The sampling accuracy for threshold generation
     */
    SieveStreaming(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);

        for (auto t : ts) {
            sieves.push_back(std::make_unique<Sieve>(K, f, t));
        }
    }

    /**
     * @brief Construct a new SieveStreaming object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state. 
     * @param m The maximum value of the singleton set, \f$ m = max_e f({e}) \f$
     * @param epsilon The sampling accuracy for threshold generation
     */
    SieveStreaming(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            sieves.push_back(std::make_unique<Sieve>(K, f, t));
        }
    }

    /**
     * @brief  Returns the number of sieves.
     * @note   
     * @retval The number of sieves.
     */
    unsigned int get_num_candidate_solutions() const {
        return sieves.size();
    }

    /**
     * @brief  Returns the total number of items stored across all sieves.
     * @note   
     * @retval The total number of items stored across all sieves.
     */
    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const & s : sieves) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

    /**
     * @brief Destroy the Sieve Streaming object
     * 
     */
    ~SieveStreaming() {
        // for (auto s : sieves) {
        //     delete s;
        // }
    }

    /**
     * @brief  Consume the next object in the data stream. This checks for each sieve if the given object exceeds the marginal gain thresholdhold and adds it to the corresponding solution.
     * 
     * @note   
     * @param  &x: A constant reference to the next object on the stream.
     * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
     * @retval None
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        for (auto &s : sieves) {
            s->next(x, id);
            if (s->get_fval() > fval) {
                fval = s->get_fval();
                // TODO THIS IS A COPY AT THE MOMENT
                solution = s->solution;
            }
        }
        is_fitted = true;
    }
};

#endif