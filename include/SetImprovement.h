#ifndef SET_IMPROVEMENT_H
#define SET_IMPROVEMENT_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <string>
#include <queue>

#include <iostream>

class SetImprovement : public SubmodularOptimizer {

protected:
    struct Pair {
        data_t weight;
        unsigned int idx;

        Pair(data_t _weight, unsigned int _idx) {
            weight = _weight;
            idx = _idx;
        }

        // bool operator > (const Pair &other) const { 
        //     return weight < other.weight; 
        // } 

        bool operator < (const Pair &other) const { 
            return weight > other.weight; 
        } 
    };

    std::priority_queue<Pair> weights; 
public:

    SetImprovement(unsigned int K, SubmodularFunction & f) : SubmodularOptimizer(K,f)  {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    SetImprovement(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : SubmodularOptimizer(K,f) {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }
    
    void next(std::vector<data_t> const &x) {
        unsigned int Kcur = solution.size();
        if (Kcur < K) {
            data_t fdelta = f->peek(solution, x, solution.size()); //- fval;
            f->update(solution, x, solution.size());
            solution.push_back(x);
            weights.push(Pair(fdelta, Kcur));
        } else {
            Pair to_replace = weights.top();
            data_t fdelta = f->peek(solution, x, to_replace.idx); // - fval;
            if (fdelta > 2*to_replace.weight) {
                //std::cout << "REPLACED" << std::endl;
                f->update(solution, x, to_replace.idx);
                solution[to_replace.idx] = x; 
                weights.pop();
                weights.push(Pair(fdelta, Kcur));
            }
        }
        fval = f->operator()(solution);
        is_fitted = true;
    }
};

#endif