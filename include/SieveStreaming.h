#ifndef SIEVESTREAMING_H
#define SIEVESTREAMING_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

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

        if (ilower >= upper)
            throw std::runtime_error("thresholds: Lower threshold boundary (" + std::to_string(ilower) + ") is higher than or equal to the upper boundary ("
                                    + std::to_string(upper) + "), epsilon = " + std::to_string(epsilon) + ".");

        for (data_t val = std::pow(1.0 + epsilon, ilower); val <= upper; ++ilower, val = std::pow(1.0 + epsilon, ilower)) {
            ts.push_back(val);
        }
    } else {
        throw std::runtime_error("thresholds: epsilon must be a positive real-number (is: " + std::to_string(epsilon) + ").");
    }
    
    return ts;
}

class SieveStreaming : public SubmodularOptimizer {
protected:

    class Sieve : public SubmodularOptimizer {
    public:
        data_t threshold;

        Sieve(unsigned int K, SubmodularFunction & f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        // Sieve(unsigned int K, std::shared_ptr<SubmodularFunction> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        Sieve(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {
        }

        void fit(std::vector<std::vector<data_t>> const & X) {
            throw std::runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;
                data_t tau = (threshold / 2.0 - fval) / static_cast<data_t>(K - Kcur);
                if (fdelta >= tau) {
                    f->update(solution, x, solution.size());
                    solution.push_back(x);
                    fval += fdelta;
                }
            }
        }

    };

protected:
    std::vector<Sieve*> sieves;

public:
    SieveStreaming(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);

        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    SieveStreaming(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    void fit(std::vector<std::vector<data_t>> const & X) {
        bool one_pass = false;
        while(solution.size() < K) {
            for (auto &x : X) {
                next(x);
                // It is verly likely that the lower threshold sieves will fill up early and thus we will probably find a full sieve early on
                // This likely results in a very bad function value. However, only iterating once over the entire data-set may lead to a very
                // weird situation where no sieve is full yet (e.g. for very small datasets). Thus, we re-iterate as often as needed and early
                // exit if we have seen every item at-least once
                if (solution.size() == K && one_pass) {
                    break;
                }
            }
            one_pass = true;
        }
    }

    ~SieveStreaming() {
        for (auto s : sieves) {
            delete s;
        }
    }

    void next(std::vector<data_t> const &x) {
        for (auto s : sieves) {
            s->next(x);
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