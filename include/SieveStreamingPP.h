#ifndef SIEVESTREAMINGPP_H
#define SIEVESTREAMINGPP_H

#include "DataTypeHandling.h"
#include "SieveStreaming.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

/**
 * @brief The SieveStreaming++ optimizer for nonnegative, monotone submodular functions. This is an improved version of SieveStreaming which re-samples thresholds once a new (better) lower bound is detected. Note that this implementation also requires that \f$ m = max_e f({e}) \f$ is known beforehand.
 * 
 *  - Stream:  Yes
 *  - Solution: \f$ 1/2 - \varepsilon \f$
 *  - Runtime: \f$ O(1) \f$
 *  - Memory: \f$ O(K / \varepsilon) \f$
 *  - Function Queries per Element: \f$ O(log(K) / \varepsilon) \f$
 *  - Function Types: nonnegative, monotone submodular functions
 * 
 * Example usage in C++:
 * @code{.cpp}
 *  //read some data 
 *  std::vector<std::vector<data_t>> = read_some_data(); 
 *  auto K = 50;
 *  // Define the function to be maximized and select the summary
 *  FastIVM fastIVM(K, RBFKernel( std::sqrt(data[0].size()), 1.0) , 1.0);
 *  SieveStreamingPP opt(K, fastIVM, 1.0, 0.1);
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
 *  opt = SieveStreamingPP(K, fastLogDet, 1.0, 0.1)
 *  opt.fit(X, K)
 *  print("fval: {} num_elements: {} num_candidates: {}".format(opt.get_fval(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions()))
 *  # process summary
 *  summary = opt.get_solution()
 * @endcode
 * 
 * @note Since this is an extension of SieveStreaming it seems likely that this class should be an extension of SieveStreaming. However, I decided against this, since the actual benefit by this is minimal. The `fit` is substantially different from SieveStreaming and the internal Sieve class uses a slightly different thresholding rule. Thus I decided to separate both implementations.  
 * 
 * __References__
 * 
 * - Kazemi, E., Mitrovic, M., Zadimoghaddam, M., Lattanzi, S., & Karbasi, A. (2019). Submodular streaming in all its glory: Tight approximation, minimum memory and low adaptive complexity. 36th International Conference on Machine Learning, ICML 2019, 2019-June, 5767–5784. Retrieved from http://proceedings.mlr.press/v97/kazemi19a/kazemi19a.pdf

*/
class SieveStreamingPP : public SubmodularOptimizer {
private:

    /**
     * @brief  A single sieve with its own threshold and accompanying summary.  
     * @note   This class is basically also implemented in SieveStreaming and - to some extend - in Salsa. I decided against a unified class for these Sieves, since the thresholding rules are often slightly different from paper to paper. I tried to stick as close as possible to the pseudocode in the papers.
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
             * @brief  Consume the next object in the data stream. This call compares the marginal gain against the given threshold and add the current item to the current solution if it exceeds the given threshold. 
             * 
             * @param  &x: A constant reference to the next object on the stream.
             * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
             */
            void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
                unsigned int Kcur = solution.size();
                if (Kcur < K) {
                    data_t fdelta = f->peek(solution, x, solution.size()) - fval;

                    if (fdelta >= threshold) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        if (id.has_value()) ids.push_back(id.value());
                        fval += fdelta;
                    }
                }
                is_fitted = true;
            }
        };    

    // The lower bound from which threshold should be sampled. This is per default 0 and will be changed when a new, better lower_bound occurs
    data_t lower_bound;

    // Maximum singleton item value
    data_t m;

    // Epsilon parameter used to sample thresholds according to the "SieveStreaming" rule
    data_t epsilon;

public:
    // The list of sieves managed by SieveStreamingPP
    std::vector<std::unique_ptr<Sieve>> sieves;

    /**
     * @brief Construct a new SieveStreamingPP object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     * @param m The maximum value of the singleton set, \f$ m = max_e f({e}) \f$
     * @param epsilon The sampling accuracy for threshold generation
     */
    SieveStreamingPP(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) 
        : SubmodularOptimizer(K,f), lower_bound(0), m(m), epsilon(epsilon) {
            // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

            // for (auto t : ts) {
            //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
            // }
        }

    /**
     * @brief Construct a new SieveStreamingPP object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state. 
     * @param m The maximum value of the singleton set, \f$ m = max_e f({e}) \f$
     * @param epsilon The sampling accuracy for threshold generation
     */
    SieveStreamingPP(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) 
        : SubmodularOptimizer(K,f), lower_bound(0), m(m), epsilon(epsilon) {
            // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

            // for (auto t : ts) {
            //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
            // }
        }

    /**
     * @brief  Returns the number of sieves.
     */
    unsigned int get_num_candidate_solutions() const {
        return sieves.size();
    }

    /**
     * @brief  Returns the total number of items stored across all sieves.
     */
    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const & s : sieves) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

    /**
     * @brief  Consume the next object in the data stream. This checks for each sieve if the given object exceeds the marginal gain threshold and adds it to the corresponding solution.
     * 
     * @param  &x: A constant reference to the next object on the stream.
     * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        if (lower_bound != fval || sieves.size() == 0) {
            lower_bound = fval;
            data_t tau_min = std::max(lower_bound, m) / static_cast<data_t>(2.0*K);
            auto no_sieves_before = sieves.size();

            auto res = std::remove_if(sieves.begin(), sieves.end(), 
                [tau_min](auto const &s) { return s->threshold < tau_min; }
            );
            sieves.erase(res, sieves.end());

            if (no_sieves_before > sieves.size() || no_sieves_before == 0) {
                std::vector<data_t> ts = thresholds(tau_min/(1.0 + epsilon), K * m, epsilon);
                
                for (auto t : ts) {
                    bool any = std::any_of(sieves.begin(), sieves.end(), 
                        [t](auto const &s){ return s->threshold == t; }
                    );
                    if (!any) {
                        sieves.push_back(std::make_unique<Sieve>(K, *f, t));
                    }
                }
            }
        }

        // std::cout << sieves.size() << std::endl;
        for (auto &s : sieves) {
            s->next(x, id);
            if (s->get_fval() > fval) {
                fval = s->get_fval();
                // TODO THIS IS A COPY AT THE MOMENT
                solution = s->solution;
            }
        }
        is_fitted = true;
    };
};

#endif