#ifndef SIEVESTREAMINGPP_H
#define SIEVESTREAMINGPP_H

#include "DataTypeHandling.h"
#include "SieveStreaming.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

class SieveStreamingPP : public SieveStreaming {
private:
    data_t lower_bound;
    data_t m;
    data_t epsilon;

public:

    SieveStreamingPP(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) : SieveStreaming(K,f, m, epsilon), lower_bound(0), m(m), epsilon(epsilon) {
    }

    SieveStreamingPP(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) : SieveStreaming(K,f, m, epsilon), lower_bound(0), m(m), epsilon(epsilon) {
    }

    void next(std::vector<data_t> const &x) {
        SieveStreaming::next(x);
        
        data_t tau_min = std::max(lower_bound, m) / static_cast<data_t>(2.0*K);
        auto res = std::remove_if(sieves.begin(), sieves.end(), 
            [tau_min](auto const s) { return s->fval < tau_min; }
        );
        sieves.erase(res, sieves.end());

        std::vector<data_t> ts = thresholds(tau_min, K * m, epsilon);
        for (auto t : ts) {
            bool any = std::any_of(sieves.begin(), sieves.end(), 
                [t](auto const s){ return s->threshold == t; }
            );
            if (!any) {
                sieves.push_back(new Sieve(K, f->clone(), t));
            }
        }
    };
};

#endif