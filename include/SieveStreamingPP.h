#ifndef SIEVESTREAMINGPP_H
#define SIEVESTREAMINGPP_H

#include "DataTypeHandling.h"
#include "SieveStreaming.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

/**
 * @brief The SieveStreaming optimizer for nonnegative, monotone submodular functions. This is an improved version of SieveStreaming which re-samples thresholds once a new (better) lower bound is detected. 
 *  - Stream:  Yes
 *  - Solution: 1/2 - \varepsilon 
 *  - Runtime: O(1)
 *  - Memory: O(K / \varepsilon)
 *  - Function Queries per Element: O(log(K) / \varepsilon)
 *  - Function Types: nonnegative, monotone submodular functions
 * 
 * See also:
 *   - Kazemi, E., Mitrovic, M., Zadimoghaddam, M., Lattanzi, S., & Karbasi, A. (2019). Submodular streaming in all its glory: Tight approximation, minimum memory and low adaptive complexity. 36th International Conference on Machine Learning, ICML 2019, 2019-June, 5767–5784. Retrieved from http://proceedings.mlr.press/v97/kazemi19a/kazemi19a.pdf
*/
class SieveStreamingPP : public SubmodularOptimizer {
private:

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
             * @param x A constant reference to the next object on the stream.
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


    data_t lower_bound;
    data_t m;
    data_t epsilon;

public:
    std::vector<std::unique_ptr<Sieve>> sieves;

    SieveStreamingPP(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) 
        : SubmodularOptimizer(K,f), lower_bound(0), m(m), epsilon(epsilon) {
            // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

            // for (auto t : ts) {
            //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
            // }
        }

    SieveStreamingPP(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) 
        : SubmodularOptimizer(K,f), lower_bound(0), m(m), epsilon(epsilon) {
            // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

            // for (auto t : ts) {
            //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
            // }
        }

    unsigned int get_num_candidate_solutions() const {
        return sieves.size();
    }

    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const & s : sieves) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

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