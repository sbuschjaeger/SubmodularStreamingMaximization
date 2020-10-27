#ifndef SIEVESTREAMING_H
#define SIEVESTREAMING_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

class SieveStreaming : public SubmodularOptimizer {
private:

    class Sieve : public SubmodularOptimizer {
    private:
        data_t threshold;
        
    public:
        
        // Sieve(unsigned int K, std::unique_ptr<SubmodularFunction> f, data_t threshold) : SubmodularOptimizer(K,std::move(f)), threshold(threshold) {
        // }

        Sieve(unsigned int K, SubmodularFunction & f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        Sieve(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {
        }

        void fit(std::vector<std::vector<data_t>> const & X) {
            throw std::runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x) - fval;
                data_t tau = (threshold / 2.0 - fval) / static_cast<data_t>(K - Kcur);
                if (fdelta >= tau) {
                    f->update(solution, x);
                    solution.push_back(std::vector<data_t>(x));
                    fval += fdelta;
                }
            }
        }

    };

    static inline std::vector<data_t> thresholds(unsigned int K, data_t m, data_t epsilon) {
        if (epsilon > 0.0) {

            int lower, upper;
            data_t tlower = std::log(m) / std::log(1.0 + epsilon);

            if (tlower > 0)
                lower = (int) tlower + 1;
            else
                lower = (int) tlower;

            data_t tupper = std::log(K * m) / std::log(1.0 + epsilon);
            upper = (int) tupper; // "+ 1" for testing

            if (lower >= upper)
                throw std::runtime_error("SieveStreaming::thresholds: Lower threshold boundary (" + std::to_string(lower) + ") is higher than or equal to the upper boundary ("
                                        + std::to_string(upper) + "), epsilon = " + std::to_string(epsilon) + ".");

            std::vector<data_t> thresholds;
            for (int i = lower; i < upper; ++i)
                thresholds.push_back(std::pow(1.0 + epsilon, i));

            return thresholds;
        } else
            throw std::runtime_error("SieveStreaming::thresholds: epsilon must be a positive real-number (is: " + std::to_string(epsilon) + ").");
    }

protected:
    // TODO How to properly init these?
    std::vector<Sieve*> sieves;


public:
    // TODO fval / get_solution wie genau implementieren?
    // TODO Sieve must recieve a copy of SumodularFunction!
    SieveStreaming(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(K, m, epsilon);
        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    // SieveStreaming(unsigned int K, std::unique_ptr<SubmodularFunction> f, data_t m, data_t epsilon) : SubmodularOptimizer(K,std::move(f)) {
    //     std::vector<data_t> ts = thresholds(K, m, epsilon);
    //     for (auto t : ts) {
    //         sieves.push_back(Sieve(K, std::move(f), t));
    //     }
    // }

    SieveStreaming(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(K, m, epsilon);
        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    ~SieveStreaming() {
        for (auto s : sieves) {
            delete s;
        }
    }

    void fit(std::vector<std::vector<data_t>> const & X) {
        for (auto &x : X) {
            next(x);
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