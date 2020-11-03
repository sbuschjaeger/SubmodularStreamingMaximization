#ifndef THREESIEVES_H
#define THREESIEVES_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <string>

class ThreeSieves : public SubmodularOptimizer {

public:
    enum THRESHOLD_STRATEGY {SIEVE,CONSTANT};

    data_t threshold;
    data_t epsilon;
    THRESHOLD_STRATEGY strategy;
    
    unsigned int T;
    unsigned int t;

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

    ThreeSieves(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon, THRESHOLD_STRATEGY strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon), strategy(strategy), T(T), t(0)  {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    ThreeSieves(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon, THRESHOLD_STRATEGY strategy, unsigned int T) : SubmodularOptimizer(K,f), threshold(K*m), epsilon(epsilon), strategy(strategy), T(T), t(0) {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    void fit(std::vector<std::vector<data_t>> const & X) {
        while(solution.size() < K) {
            for (auto &x : X) {
                next(x);
                if (solution.size() == K) {
                    break;
                }
            }
        }
    }

    void next(std::vector<data_t> const &x) {
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