
.. _program_listing_file_include_SieveStreamingPP.h:

Program Listing for File SieveStreamingPP.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_SieveStreamingPP.h>` (``include/SieveStreamingPP.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef SIEVESTREAMINGPP_H
   #define SIEVESTREAMINGPP_H
   
   #include "DataTypeHandling.h"
   #include "SieveStreaming.h"
   #include <vector>
   #include <algorithm>
   #include <numeric>
   #include <unordered_set>
   
   class SieveStreamingPP : public SubmodularOptimizer {
   private:
   
       class Sieve : public SubmodularOptimizer {
           public:
               // The threshold
               data_t threshold;
   
               Sieve(unsigned int K, SubmodularFunction & f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}
   
               Sieve(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {
               }
   
               void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
                   throw std::runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
               }
   
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
