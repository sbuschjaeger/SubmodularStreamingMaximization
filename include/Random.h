#ifndef RANDOM_H
#define RANDOM_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

class Random : public SubmodularOptimizer {
protected:
    unsigned int cnt = 0;
    std::default_random_engine generator;

    // Taken from https://www.gormanalysis.com/blog/random-numbers-in-cpp/#sampling-without-replacement
    static inline std::vector<unsigned int> sample_without_replacement(int k, int N, std::default_random_engine& gen) {
        // Sample k elements from the range [1, N] without replacement
        // k should be <= N
        
        // Create an unordered set to store the samples
        std::unordered_set<unsigned int> samples;
        
        // Sample and insert values into samples
        for (int r = N - k; r < N; ++r) {
            unsigned int v = std::uniform_int_distribution<>(1, r)(gen);
            if (!samples.insert(v).second) samples.insert(r - 1);
        }
        
        // Copy samples into vector
        std::vector<unsigned int> result(samples.begin(), samples.end());
        
        // Shuffle vector
        std::shuffle(result.begin(), result.end(), gen);
        
        return result;
    };

public:
    Random(unsigned int K, SubmodularFunction & f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}

    Random(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}

    void fit(std::vector<std::vector<data_t>> const & X) {
        if (X.size() < K) {
            K = X.size();
        }
        std::vector<unsigned int> indices = sample_without_replacement(K, X.size(), generator);

        for (auto i : indices) {
            f->update(solution, X[i], solution.size());
            solution.push_back(X[i]);
            //solution.push_back(std::vector<data_t>(X[i]));
        }

        cnt = X.size();
        fval = f->operator()(solution);
        is_fitted = true;
    }

    void next(std::vector<data_t> const &x) {
        // Super basic reservoir sampling. This can probably be improved.
        if (solution.size() < K) {
            f->update(solution, x, solution.size());
            solution.push_back(x);
        } else {
            unsigned int j = std::uniform_int_distribution<>(1, cnt)(generator);
            if (j <= K) {
                f->update(solution, x, j - 1);
                solution[j - 1] = x; //std::vector<data_t>(x);
            }
        }

        fval = f->operator()(solution);
        is_fitted = true;
        ++cnt;
    }
};

#endif // RANDOM_H
