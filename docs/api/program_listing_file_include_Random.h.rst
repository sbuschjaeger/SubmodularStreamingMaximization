
.. _program_listing_file_include_Random.h:

Program Listing for File Random.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_Random.h>` (``include/Random.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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
   
       static inline std::vector<idx_t> sample_without_replacement(int k, int N, std::default_random_engine& gen) {
           // Create an unordered set to store the samples
           std::unordered_set<idx_t> samples;
           
           // Sample and insert values into samples
           for (int r = N - k; r < N; ++r) {
               idx_t v = std::uniform_int_distribution<>(1, r)(gen);
               if (!samples.insert(v).second) samples.insert(r - 1);
           }
           
           // Copy samples into vector
           std::vector<idx_t> result(samples.begin(), samples.end());
           
           // Shuffle vector
           std::shuffle(result.begin(), result.end(), gen);
           
           return result;
       };
   
   public:
   
       Random(unsigned int K, SubmodularFunction & f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}
   
       Random(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, unsigned long seed = 0) : SubmodularOptimizer(K,f), generator(seed) {}
   
       void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
           if (X.size() < K) {
               K = X.size();
           }
           std::vector<idx_t> indices = sample_without_replacement(K, X.size(), generator);
   
           for (auto i : indices) {
               f->update(solution, X[i], solution.size());
               solution.push_back(X[i]);
               if (ids.size() >= i) {
                   this->ids.push_back(ids[i]);
               }
               //solution.push_back(std::vector<data_t>(X[i]));
           }
   
           cnt = X.size();
           fval = f->operator()(solution);
           is_fitted = true;
       }
   
       void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
           std::vector<idx_t> ids;
           fit(X,ids,iterations);
       }
   
       void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) {
           if (solution.size() < K) {
               // Just add the first K elements
               f->update(solution, x, solution.size());
               solution.push_back(x);
               if (id.has_value()) ids.push_back(id.value());
           } else {
               // Sample the replacement-index with decreasing probability
               unsigned int j = std::uniform_int_distribution<>(1, cnt)(generator);
               if (j <= K) {
                   f->update(solution, x, j - 1);
                   if (id.has_value()) ids[j-1] = id.value();
                   solution[j - 1] = x; 
               }
           }
   
           // Update the current function value
           fval = f->operator()(solution);
           is_fitted = true;
           ++cnt;
       }
   };
   
   #endif // RANDOM_H
