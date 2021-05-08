#ifndef THREESIEVES_H
#define THREESIEVES_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <string>

/**
 * @brief  The ThreeSieves algorithm for submodular function maximization. This optimizer tries to estimate the probability that a given item is not `out-valued' in the future. To do so, it compares the marginal gain of each item against a pre-computed threshold. If this threshold is too large and the algorithm therefore rejects most items, it reduces the threshold after \f$ T \f$ tries. The confidence interval of not finding an element in the stream which would out-value the current threshold is given by the Rule Of Three, hence the name:
 *  - Stream:  Yes
 *  - Solution: \f$ (1-\varepsilon)(1-1/\exp(1)) with probability (1-\alpha)^K \f$
 *  - Runtime: \f$ O(N) \f$
 *  - Memory: \f$ O(K) \f$
 *  - Function Queries per Element: \f$ O(1) \f$
 *  - Function Types: nonnegative submodular functions
 * 
 * Internally, this algorithm also uses a novelty threshold similar to SieveStreaming, but only maintains a single sieve. After T unsuccessful tries of adding an element to the summary, it reduces the threshold. This can either be a constant value (CONSTANT strategy) or by using the next smallest threshold from \f$ {(1+epsilon)^i  | i \in Z, lower \le (1+epsilon)^i \le upper} \f$ similar to SieveStreaming (SIEVE strategy). 
 * 
 * __References__:
 * 
 * [1] The paper is currently under review.
 * 
 * @note   
 * @retval None
 */
class ThreeSieves : public SubmodularOptimizer {

public:
    /**
     * @brief  The different strategies to reduce the threshold after T unsuccessful tries.
     *          SIEVE: Start with the largest threshold in \f$ {(1+epsilon)^i  | i \in Z, lower \le (1+epsilon)^i \le upper} \f$ and use the next largest as the new threshold
     *          CONSTANT: Reduce the threshold by a constant epsilon 
     * @note   
     * @retval None
     */
    enum THRESHOLD_STRATEGY {SIEVE,CONSTANT};

    // the current threshold
    data_t threshold;

    // the epsilon parameter for the threshold strategy 
    data_t epsilon;

    // the actual threshold strategy
    THRESHOLD_STRATEGY strategy;
    
    // maximum number of tries
    unsigned int T;

    // current number of tries
    unsigned int t;

    /**
     * @brief  Construct a new ThreeSieves object
     * @note   
     * @param  K: The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param  f: The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
     * @param  m: The maximum singleton value 
     * @param  epsilon: The epsilon parameter used in the thresholding strategy
     * @param  strategy: The thresholding strategy. Uses SIEVE if "sieve" (or any lower/upper-case variation) is supplied. Otherwise uses CONSTANT
     * @param  T: The maximum number of tries until the threshold is reduced
     * @retval The newly constructed object.
     */
    ThreeSieves(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon, std::string const & strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon),T(T), t(0)  {
        // assert(("T should at-least be 1 or greater.", T >= 1));
        std::string lower_case(strategy);
        std::transform(lower_case.begin(), lower_case.end(), lower_case.begin(),
            [](unsigned char c){ return std::tolower(c); });
        
        if (lower_case == "sieve") {
            this->strategy = THRESHOLD_STRATEGY::SIEVE;
        } else {
            this->strategy = THRESHOLD_STRATEGY::CONSTANT;
        }
    }
    
    /**
     * @brief  Construct a new ThreeSieves object
     * @note   
     * @param  K: The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param  f: The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     * @param  m: The maximum singleton value 
     * @param  epsilon: The epsilon parameter used in the thresholding strategy
     * @param  strategy: The thresholding strategy. Uses SIEVE if "sieve" (or any lower/upper-case variation) is supplied. Otherwise uses CONSTANT
     * @param  T: The maximum number of tries until the threshold is reduced
     * @retval The newly constructed object.
     */
    ThreeSieves(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon, std::string const & strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon), T(T), t(0) {
        std::string lower_case(strategy);
        std::transform(lower_case.begin(), lower_case.end(), lower_case.begin(),
            [](unsigned char c){ return std::tolower(c); });

        if (lower_case == "sieve") {
            this->strategy = THRESHOLD_STRATEGY::SIEVE;
        } else {
            this->strategy = THRESHOLD_STRATEGY::CONSTANT;
        }
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    /**
     * @brief  Construct a new ThreeSieves object
     * @note   
     * @param  K: The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param  f: The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
     * @param  m: The maximum singleton value 
     * @param  epsilon: The epsilon parameter used in the thresholding strategy
     * @param  strategy: The thresholding strategy. 
     * @param  T: The maximum number of tries until the threshold is reduced
     * @retval The newly constructed object.
     */
    ThreeSieves(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon, THRESHOLD_STRATEGY strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon), strategy(strategy), T(T), t(0)  {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    /**
     * @brief  Construct a new ThreeSieves object
     * @note   
     * @param  K: The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param  f: The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     * @param  m: The maximum singleton value 
     * @param  epsilon: The epsilon parameter used in the thresholding strategy
     * @param  strategy: The thresholding strategy.
     * @param  T: The maximum number of tries until the threshold is reduced
     * @retval The newly constructed object.
     */
    ThreeSieves(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon, THRESHOLD_STRATEGY strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon), strategy(strategy), T(T), t(0) {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }
    
    /**
     * @brief  Consume the next object in the data stream. If more than T tries have already been performed, then the threshold is lower according to the threshold strategy. In any case, the current item's marginal gain is compared to the current / changed threshold and the item is added if the gain exceeds it. If so, the summary is updated accordingly
     * 
     * @note   
     * @param  &x: A constant reference to the next object on the stream.
     * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
     * @retval None
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        unsigned int Kcur = solution.size();
        if (Kcur < K) {
            if (t >= T) {
                switch(strategy) {
                    case THRESHOLD_STRATEGY::SIEVE: 
                    {
                        data_t tmp = std::log(threshold) / std::log(1.0 + epsilon);
                        int i;
                        if (tmp == std::floor(tmp) || std::abs(tmp - std::floor(tmp)) < 1e-7) {
                            i = std::floor(tmp) - 1;
                        } else {
                            i = std::floor(tmp);
                        }
                        threshold = std::pow(1+epsilon, i);
                        break;
                    }
                    case THRESHOLD_STRATEGY::CONSTANT:
                    {
                        threshold -= threshold - epsilon;
                        break;
                    }
                }
                t = 0;
            }

            data_t fdelta = f->peek(solution, x, solution.size()) - fval;
            data_t tau = (threshold / 2.0 - fval) / static_cast<data_t>(K - Kcur);
            
            if (fdelta >= tau) {
                f->update(solution, x, solution.size());
                solution.push_back(x);
                if (id.has_value()) ids.push_back(id.value());
                fval += fdelta;
                t = 0;
            } else {
                ++t;
            }
        }
        is_fitted = true;
    }
};

#endif