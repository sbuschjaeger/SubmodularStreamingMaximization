#ifndef SALSA_H
#define SALSA_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include "SieveStreaming.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

class Salsa : public SubmodularOptimizer {
protected:

    class FixedThreshold : public SubmodularOptimizer {
    private:
        data_t epsilon;
        data_t threshold;

    public:
        FixedThreshold(unsigned int K, SubmodularFunction & f, data_t epsilon, data_t threshold) 
            : SubmodularOptimizer(K,f), epsilon(epsilon), threshold(threshold) {}

        FixedThreshold(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t epsilon, data_t threshold) 
            : SubmodularOptimizer(K,f),  epsilon(epsilon), threshold(threshold) {}

        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("FixedThresholds are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;
                
                if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 + epsilon))) {
                    f->update(solution, x, solution.size());
                    solution.push_back(x);
                    fval += fdelta;
                }
            }
            is_fitted = true;
        }
    };

    class Dense : public SubmodularOptimizer {
    private:
        data_t threshold;
        data_t beta;
        data_t C1;
        data_t C2;
        unsigned int N;
        unsigned int observed;

    public:
        Dense(unsigned int K, SubmodularFunction & f, data_t threshold, data_t beta, data_t C1, data_t C2, unsigned int N) 
            : SubmodularOptimizer(K,f), threshold(threshold), beta(beta), C1(C1), C2(C2), N(N), observed(0) {}

        Dense(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold, data_t beta, data_t C1, data_t C2, unsigned int N) 
            : SubmodularOptimizer(K,f), threshold(threshold), beta(beta), C1(C1), C2(C2), N(N), observed(0) {}

        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("FixedThresholds are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;

                if (static_cast<data_t>(observed) <= beta * static_cast<data_t>(N)) {
                    // First threshold
                    if (fdelta >= (C1 * threshold) / static_cast<data_t>(K)) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        fval += fdelta;
                    }
                } else {
                    // Second threshold
                    if (fdelta >= threshold / (C2 * static_cast<data_t>(K))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        fval += fdelta;
                    }
                }
            }
            observed++;
            is_fitted = true;
        }
    };

    class HighLowThreshold : public SubmodularOptimizer {
    private:
        data_t epsilon;
        data_t threshold;
        data_t beta;
        data_t delta;
        unsigned int N;
        unsigned int observed;

    public:
        HighLowThreshold(unsigned int K, SubmodularFunction & f, data_t epsilon, data_t threshold, data_t beta, data_t delta, unsigned int N) 
            : SubmodularOptimizer(K,f), epsilon(epsilon), threshold(threshold), beta(beta), delta(delta), N(N), observed(0) {}

        HighLowThreshold(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t epsilon, data_t threshold, data_t beta, data_t delta, unsigned int N) 
            : SubmodularOptimizer(K,f),  epsilon(epsilon), threshold(threshold), beta(beta), delta(delta), N(N), observed(0) {}

        void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
            throw std::runtime_error("HighLowThreshold are only meant to be used through Salsa and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;

                if (static_cast<data_t>(observed) <= beta * static_cast<data_t>(N)) {
                    // High threshold
                    if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 + epsilon))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        fval += fdelta;
                    }
                } else {
                    // Low threshold
                    if (fdelta >= ((threshold / static_cast<data_t>(K)) * (0.5 - delta))) {
                        f->update(solution, x, solution.size());
                        solution.push_back(x);
                        fval += fdelta;
                    }
                }
            }
            observed++;
            is_fitted = true;
        }
    };
    

protected:
    std::vector<std::unique_ptr<SubmodularOptimizer>> algos;
    data_t m;
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

    unsigned int get_num_candidate_solutions() const {
        return algos.size();
    }

    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const & s : algos) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

    void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        unsigned int N = X.size();
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            algos.push_back(std::make_unique<FixedThreshold>(K, *f, fixed_epsilon, t));
            algos.push_back(std::make_unique<HighLowThreshold>(K, *f, hilow_epsilon, t, hilow_beta, hilow_delta, N));
            algos.push_back(std::make_unique<Dense>(K, *f, t, dense_beta, dense_C1, dense_C2, N));
        }

        for (unsigned int i = 0; i < iterations; ++i) {
            for (auto &x : X) {
                for (auto &s : algos) {
                    s->next(x);
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

    void next(std::vector<data_t> const &x) {
        throw std::runtime_error("Salsa does not support streaming data, please use fit().");
    }
};

#endif