#ifndef SALSA_H
#define SALSA_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include "SieveStreaming.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

/**
 * @brief  The Salsa optimizer for submodular functions. This algorithms runs multiple copies of different thresholding strategies in parallel. Some of these strategies require additional information about the datastream such as its length and thus this algorithm might not be applicable in a "real" streaming scenario
 *  - Stream:  (Yes)
 *  - Solution: \f$1/2 - \varepsilon\f$
 *  - Runtime: \f$O(1)\f$
 *  - Memory: \f$O(K \cdot log(K) / \varepsilon)\f$
 *  - Function Queries per Element: \f$O(log(K) / \varepsilon)\f$
 *  - Function Types: nonnegative, monotone submodular functions
 * 
 * Example usage in C++:
 * @code{.cpp}
 *  //read some data 
 *  std::vector<std::vector<data_t>> = read_some_data(); 
 *  auto K = 50;
 *  // Define the function to be maximized and select the summary
 *  FastIVM fastIVM(K, RBFKernel( std::sqrt(data[0].size()), 1.0) , 1.0);
 *  Salsa opt(K, fastIVM, 1.0, 0.01);
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
 *  opt = Salsa(K, fastLogDet, 1.0, 0.01)
 *  opt.fit(X, K)
 *  print("fval: {} num_elements: {} num_candidates: {}".format(opt.get_fval(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions()))
 *  # process summary
 *  summary = opt.get_solution()
 * @endcode
 * 
 * __References__
 * 
 * - Norouzi-Fard, A., Tarnawski, J., Mitrovic, S., Zandieh, A., Mousavifar, A. & Svensson, O.. (2018). Beyond 1/2-Approximation for Submodular Maximization on Massive Data Streams. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:3829-3838 
 * - Norouzi-Fard, A., Tarnawski, J., Mitrovic, S., Zandieh, A., Mousavifar, A. & Svensson, O.. (2018). Beyond 1/2-Approximation for Submodular Maximization on Massive Data Streams. https://arxiv.org/abs/1808.01842
 */
class Salsa : public SubmodularOptimizer {
protected:

    /**
     * @brief  Fixed thresholding strategy (Algorithm 2 in the ICML paper). This basically simulates the thresholding strategy of SieveStreaming with a slightly different sampling strategy for the thresholds. In the original version OPT is known and different epsilon are used to "sample" different thresholds. As detailed in the longer version arxiv of the paper, we can estimate OPT via \f$O = \{(1+\varepsilon)^i \mid i \in \mathbb{Z}, m \le (1+\varepsilon)^i \le K \cdot m\}\f$ where \f$ m = \max f({m}) \f$ is the maximum singleton function value. 
     * @note   This class is basically also implemented in SieveStreaming and SieveStreamingPP. I decided against a unified class for these Sieves, since the thresholding rules are often slightly different from paper to paper. I tried to stick as close as possible to the pseudocode in the papers.
     */
    class FixedThreshold : public SubmodularOptimizer {
    private:

        // Epsilon parameter
        data_t epsilon;

        // Sampled OPT threshold 
        data_t threshold;

    public:

        /**
         * @brief Construct a new FixedThreshold object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
         * @param epsilon The epsilon parameter for this algorithm
         * @param threshold The (sampled) OPT threshold  
         */
        FixedThreshold(unsigned int K, SubmodularFunction & f, data_t epsilon, data_t threshold) 
            : SubmodularOptimizer(K,f), epsilon(epsilon), threshold(threshold) {}


        /**
         * @brief Construct a new FixedThreshold object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
         * @param epsilon The epsilon parameter for this algorithm
         * @param threshold The (sampled) OPT threshold
         */
        FixedThreshold(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t epsilon, data_t threshold) 
            : SubmodularOptimizer(K,f),  epsilon(epsilon), threshold(threshold) {}

        /**
         * @brief Throws an exception when called. FixedThreshold should not be used outside Salsa.
         * 
         * @param X A constant reference to the entire data set
         * @param iterations: Number of iterations over the entire dataset
         */
        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("FixedThresholds are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        /**
         * @brief  Consume the next object in the data stream. Add the current item to the summary if there are fewer than K element in it and if the items gain exceeds the current thresholding rule. Performs one function query.
         * 
         * @param  &x: A constant reference to the next object on the stream.
         * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
         */
        void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;
                
                if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 + epsilon))) {
                    f->update(solution, x, solution.size());
                    solution.push_back(x);
                    if (id.has_value()) ids.push_back(id.value());
                    fval += fdelta;
                }
            }
            is_fitted = true;
        }
    };

    /**
     * @brief  Dense thresholding strategy (Algorithm 1 in the ICML paper). This basically simulates the SimpleGreedy / PreemptionStreaming algalgorithm with a slightly different sampling strategy for the thresholds. In the original version OPT is known and different epsilon are used to "sample" different thresholds. As detailed in the longer version arxiv version of the paper, we can estimate OPT via \f$O = \{(1+\varepsilon)^i \mid i \in \mathbb{Z}, m \le (1+\varepsilon)^i \le K \cdot m\}\f$ where \f$ m = \max f({m}) \f$ is the maximum singleton function value. 
     */
    class Dense : public SubmodularOptimizer {
    private:
        // Sampled OPT threshold 
        data_t threshold;

        // Additional hyper parameters
        data_t beta;
        data_t C1;
        data_t C2;

        // Total number of items in the datastream
        unsigned int N;

        // Total number of observed items so far
        unsigned int observed;

    public:
    
        /**
         * @brief Construct a new Dense object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
         * @param  threshold: The (sampled) OPT threshold
         * @param  beta: The \f$\beta\f$ parameter
         * @param  C1: The \f$\C_1\f$ parameter
         * @param  C2: The \f$C_2\f$ parameter
         * @param  N: The number of items in the datastream
         */
        Dense(unsigned int K, SubmodularFunction & f, data_t threshold, data_t beta, data_t C1, data_t C2, unsigned int N) 
            : SubmodularOptimizer(K,f), threshold(threshold), beta(beta), C1(C1), C2(C2), N(N), observed(0) {}

        /**
         * @brief Construct a new Dense object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
         * @param  threshold: The (sampled) OPT threshold
         * @param  beta: The \f$\beta\f$ parameter
         * @param  C1: The \f$\C_1\f$ parameter
         * @param  C2: The \f$C_2\f$ parameter
         * @param  N: The number of items in the datastream
         */
        Dense(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold, data_t beta, data_t C1, data_t C2, unsigned int N) 
            : SubmodularOptimizer(K,f), threshold(threshold), beta(beta), C1(C1), C2(C2), N(N), observed(0) {}

        /**
         * @brief Throws an exception when called. FixedThreshold should not be used outside Salsa.
         * 
         * @param X A constant reference to the entire data set
         * @param iterations: Number of iterations over the entire dataset
         */
        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("FixedThresholds are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        /**
         * @brief  Consume the next object in the data stream. Add the current item to the summary if there are fewer than K element in it and if the items gain exceeds the current thresholding rule. Performs one function query.
         * 
         * @param  &x: A constant reference to the next object on the stream.
         * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
         */
        void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;

                if (static_cast<data_t>(observed) <= beta * static_cast<data_t>(N)) {
                    // First threshold
                    if (fdelta >= (C1 * threshold) / static_cast<data_t>(K)) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        if (id.has_value()) ids.push_back(id.value());
                        fval += fdelta;
                    }
                } else {
                    // Second threshold
                    if (fdelta >= threshold / (C2 * static_cast<data_t>(K))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        if (id.has_value()) ids.push_back(id.value());
                        fval += fdelta;
                    }
                }
            }
            observed++;
            is_fitted = true;
        }
    };

    /**
     * @brief  High-Low thresholding strategy (Algorithm 3 in in the ICML paper]). This basically combines Dense and FixedThresholding. In the original version OPT is known and different epsilon are used to "sample" different thresholds. As detailed in the longer arxiv version of the paper, we can estimate OPT via \f$O = \{(1+\varepsilon)^i \mid i \in \mathbb{Z}, m \le (1+\varepsilon)^i \le K \cdot m\}\f$ where \f$ m = \max f({m}) \f$ is the maximum singleton function value. 
     */
    class HighLowThreshold : public SubmodularOptimizer {
    private:
        // Sampled OPT threshold 
        data_t threshold;

        // Additional hyper parameters
        data_t epsilon;
        data_t beta;
        data_t delta;

        // Total number of items in the datastream
        unsigned int N;

        // Total number of observed items so far
        unsigned int observed;

    public:

        /**
         * @brief Construct a new HighLowThreshold object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
         * @param  epsilon: The \f$\epsilon\f$ parameter
         * @param  threshold: The (sampled) OPT threshold
         * @param  beta: The \f$\beta\f$ parameter
         * @param  delta: The \f$\delta\f$ parameter
         * @param  N: The number of items in the datastream
         */
        HighLowThreshold(unsigned int K, SubmodularFunction & f, data_t epsilon, data_t threshold, data_t beta, data_t delta, unsigned int N) 
            : SubmodularOptimizer(K,f), epsilon(epsilon), threshold(threshold), beta(beta), delta(delta), N(N), observed(0) {}

        /**
         * @brief Construct a new HighLowThreshold object
         * 
         * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
         * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
         * @param  epsilon: The \f$\epsilon\f$ parameter
         * @param  threshold: The (sampled) OPT threshold
         * @param  beta: The \f$\beta\f$ parameter
         * @param  delta: The \f$\delta\f$ parameter
         * @param  N: The number of items in the datastream
         */
        HighLowThreshold(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t epsilon, data_t threshold, data_t beta, data_t delta, unsigned int N) 
            : SubmodularOptimizer(K,f),  epsilon(epsilon), threshold(threshold), beta(beta), delta(delta), N(N), observed(0) {}

        /**
         * @brief Throws an exception when called. FixedThreshold should not be used outside Salsa.
         * 
         * @param X A constant reference to the entire data set
         * @param iterations: Number of iterations over the entire dataset
         */
        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("HighLowThreshold are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        /**
         * @brief  Consume the next object in the data stream. Add the current item to the summary if there are fewer than K element in it and if the items gain exceeds the current thresholding rule. Performs one function query.
         * 
         * @param  &x: A constant reference to the next object on the stream.
         * @param  id: The id of the given object. If this is a `std::nullopt` this parameter is ignored. Otherwise the id is inserted into the solution. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks.  
         */
        void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;

                if (static_cast<data_t>(observed) <= beta * static_cast<data_t>(N)) {
                    // High threshold
                    if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 + epsilon))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        if (id.has_value()) ids.push_back(id.value());
                        fval += fdelta;
                    }
                } else {
                    // Low threshold
                    if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 - delta))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        if (id.has_value()) ids.push_back(id.value());
                        fval += fdelta;
                    }
                }
            }
            observed++;
            is_fitted = true;
        }
    };
    

protected:
    // List of all algorithm which are run in parallel
    std::vector<std::unique_ptr<SubmodularOptimizer>> algos;

    // Maximum singleton item value
    data_t m;

    // Epsilon parameter used to sample thresholds according to the "SieveStreaming" rule
    data_t epsilon;

    // HighLowThreshold
    data_t hilow_epsilon;
    data_t hilow_beta;
    data_t hilow_delta;

    // Dense
    data_t dense_beta;
    data_t dense_C1;
    data_t dense_C2;

    //FixedThreshold
    data_t fixed_epsilon;
public:

    /**
     * @brief  Construct a new Salsa object
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
     * @param  m: The maximum singleton function value \f$ m = \max f({m}) \f$ .
     * @param  epsilon: The \f$\varepsilon\f$ parameter used to sample different thresholds via the "SieveStreaming" rule \f$O = \{(1+\varepsilon)^i \mid i \in \mathbb{Z}, m \le (1+\varepsilon)^i \le K \cdot m\}\f$ where \f$m = \max f({m})\f$ is the maximum singleton function value.
     * @param  hilow_epsilon: The \f$\epsilon\f$ parameter of the High-Low thresholding algorithm
     * @param  hilow_beta: The \f$\beta\f$ parameter of the High-Low thresholding algorithm
     * @param  hilow_delta: The \f$\delta\f$ parameter of the High-Low thresholding algorithm
     * @param  dense_beta: The \f$\beta\f$ parameter of the Dense thresholding algorithm
     * @param  dense_C1: The \f$C_1\f$ parameter of the Dense thresholding algorithm
     * @param  dense_C2: The \f$C_2\f$ parameter of the Dense thresholding algorithm
     * @param  fixed_epsilon: The \f$\epsilon\f$ parameter of the Fixed thresholding algorithm
     */
    Salsa(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon,
        data_t hilow_epsilon = 0.05,
        data_t hilow_beta = 0.1,
        data_t hilow_delta = 0.025,
        data_t dense_beta = 0.8,
        data_t dense_C1 = 10,
        data_t dense_C2 = 0.2,
        data_t fixed_epsilon = 1.0 / 6.0
    ) : SubmodularOptimizer(K,f), 
        m(m),epsilon(epsilon), 
        hilow_epsilon(hilow_epsilon),
        hilow_beta(hilow_beta),
        hilow_delta(hilow_delta),
        dense_beta(dense_beta),
        dense_C1(dense_C1),
        dense_C2(dense_C2),
        fixed_epsilon(fixed_epsilon)
    {}

    /**
     * @brief  Construct a new Salsa object
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     * @param  m: The maximum singleton function value \f$ m = \max f({m}) \f$.
     * @param  epsilon: The \f$\varepsilon\f$ parameter used to sample different thresholds via the "SieveStreaming" rule \f$O = \{(1+\varepsilon)^i \mid i \in \mathbb{Z}, m \le (1+\varepsilon)^i \le K \cdot m\}\f$ where \f$m = \max f({m})\f$ is the maximum singleton function value.
     * @param  hilow_epsilon: The \f$\epsilon\f$ parameter of the High-Low thresholding algorithm
     * @param  hilow_beta: The \f$\beta\f$ parameter of the High-Low thresholding algorithm
     * @param  hilow_delta: The \f$\delta\f$ parameter of the High-Low thresholding algorithm
     * @param  dense_beta: The \f$\beta\f$ parameter of the Dense thresholding algorithm
     * @param  dense_C1: The \f$C_1\f$ parameter of the Dense thresholding algorithm
     * @param  dense_C2: The \f$C_2\f$ parameter of the Dense thresholding algorithm
     * @param  fixed_epsilon: The \f$\epsilon\f$ parameter of the Fixed thresholding algorithm
     */
    Salsa(unsigned int K, 
        std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon,
        data_t hilow_epsilon = 0.05,
        data_t hilow_beta = 0.1,
        data_t hilow_delta = 0.025,
        data_t dense_beta = 0.8,
        data_t dense_C1 = 10,
        data_t dense_C2 = 0.2,
        data_t fixed_epsilon = 1.0 / 6.0
    ) : SubmodularOptimizer(K,f), 
        m(m),epsilon(epsilon),
        hilow_epsilon(hilow_epsilon),
        hilow_beta(hilow_beta),
        hilow_delta(hilow_delta),
        dense_beta(dense_beta),
        dense_C1(dense_C1),
        dense_C2(dense_C2),
        fixed_epsilon(fixed_epsilon)
    {}

    /**
     * @brief  Returns the number of thresholding algorithms used in parallel. Each algorithm stores at most one full summary.
     */
    unsigned int get_num_candidate_solutions() const {
        return algos.size();
    }

    /**
     * @brief  Returns the total number of items stored across all algorithms.
     */
    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const & s : algos) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

    /**
     * @brief Executes all different thresholding algorithm in parallel and picks that one with the best summary. 
     * 
     * @param X A constant reference to the entire data set
     * @param ids: A list of identifier for each object. This can be used to uniquely identify the objects in the summary. If ids.size() < X.size(), then only partial ids are stored. No ids are stored if ids is empty. Make sure, that either _all_ or _no_ object receives an id to keep track which id belongs to which object. This algorithm simply stores the objects and the ids in two separate lists and performs no safety checks. 
     * @param iterations: Has no effect. Greedy iterates K times over the entire dataset in any case.
     */
    void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
        unsigned int N = X.size();
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            algos.push_back(std::make_unique<FixedThreshold>(K, *f, fixed_epsilon, t));
            algos.push_back(std::make_unique<HighLowThreshold>(K, *f, hilow_epsilon, t, hilow_beta, hilow_delta, N));
            algos.push_back(std::make_unique<Dense>(K, *f, t, dense_beta, dense_C1, dense_C2, N));
        }

        for (unsigned int i = 0; i < iterations; ++i) {
            for (unsigned int j = 0; j < X.size(); ++j) {
            //for (auto &x : X) {
                for (auto &s : algos) {
                    if (ids.size() == X.size()) {
                        s->next(X[j], ids[j]);
                    } else {
                        s->next(X[j]);
                    }
                    if (s->get_fval() > fval) {
                        fval = s->get_fval();
                        // TODO THIS IS A COPY AT THE MOMENT
                        solution = s->solution;
                        is_fitted = true;
                    }
                    
                    if (solution.size() == K && i > 0) {
                        return;
                    }
                }
            }
        }
    }

    /**
     * @brief Executes all different thresholding algorithm in parallel and picks that one with the best summary. 
     * @note This internally calls fit with an empty id set.
     * @param X A constant reference to the entire data set
     * @param iterations: Has no effect. Greedy iterates K times over the entire dataset in any case.
     */
    void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        std::vector<idx_t> ids;
        fit(X,ids,iterations);
    }

    /**
     * @brief Throws an exception when called. Salsa does not support streaming!
     * 
     * @param x A constant reference to the next object on the stream.
     */
    void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
        throw std::runtime_error("Salsa does not support streaming data, please use fit().");
    }
};

#endif