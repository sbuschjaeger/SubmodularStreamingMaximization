
.. _program_listing_file_include_SubmodularOptimizer.h:

Program Listing for File SubmodularOptimizer.h
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_SubmodularOptimizer.h>` (``include/SubmodularOptimizer.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef SUBMODULAROPTIMIZER_H
   #define SUBMODULAROPTIMIZER_H
   
   #include <memory>
   #include <stdexcept>
   #include <utility>
   #include <vector>
   #include <functional>
   #include <cassert>
   #include <memory>
   #include <optional>
   
   #include "SubmodularFunction.h"
   
   class SubmodularOptimizer {
   private:
       
   protected:
       // The cardinality constraint you of the optimization problem, that is the number of items selected.
       unsigned int K;
       
       //std::unique_ptr<SubmodularFunction> f;
       std::shared_ptr<SubmodularFunction> f;
   
       // true if fit() or next() has been called.
       bool is_fitted;
   
   public:
       // The current solution of this optimizer
       std::vector<std::vector<data_t>> solution;
       std::vector<idx_t> ids;
   
       // The current function value of this optimizer
       data_t fval;
   
       SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
           : K(K), f(f.clone()) {
           is_fitted = false;
           fval = 0;
           // assert(("K should at-least be 1 or greater.", K >= 1));
       }
   
       SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
           : K(K), f(std::unique_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f))) {
           is_fitted = false;
           fval = 0;
           // assert(("K should at-least be 1 or greater.", K >= 1));
       }
   
       virtual void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
           assert(X.size() == ids.size());
   
           for (unsigned int i = 0; i < iterations; ++i) {
               for (unsigned int j = 0; j < X.size(); ++j) {
                   next(X[j], ids[j]);
                   // It is very likely that the lower threshold sieves will fill up early and thus we will probably find a full sieve early on
                   // This likely results in a very bad function value. However, only iterating once over the entire data-set may lead to a very
                   // weird situation where no sieve is full yet (e.g. for very small datasets). Thus, we re-iterate as often as needed and early
                   // exit if we have seen every item at-least once
                   if (solution.size() == K && i > 0) {
                       return;
                   }
               }
           }
       }
   
       virtual void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
           for (unsigned int i = 0; i < iterations; ++i) {
               for (auto &x : X) {
                   next(x);
                   // It is very likely that the lower threshold sieves will fill up early and thus we will probably find a full sieve early on
                   // This likely results in a very bad function value. However, only iterating once over the entire data-set may lead to a very
                   // weird situation where no sieve is full yet (e.g. for very small datasets). Thus, we re-iterate as often as needed and early
                   // exit if we have seen every item at-least once
                   if (solution.size() == K && i > 0) {
                       return;
                   }
               }
           }
       }
   
       virtual void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) = 0;
   
   
       std::vector<std::vector<data_t>>const & get_solution() const {
           if (!this->is_fitted) {
                throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_solution()");
           } else {
               return solution;
           }
       }
       
       std::vector<idx_t> const &get_ids() const {
           if (!this->is_fitted) {
                throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_ids()");
           } else {
               return ids;
           }
       }
   
       virtual unsigned int get_num_candidate_solutions() const {
           return 1;
       }
       
       virtual unsigned long get_num_elements_stored() const {
           return this->get_solution().size();
       }
   
       data_t get_fval() const {
           return fval;
       }
   
       virtual ~SubmodularOptimizer() {}
   };
   
   #endif // THREESIEVES_SUBMODULAROPTIMIZER_H
