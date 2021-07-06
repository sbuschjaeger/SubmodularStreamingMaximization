
.. _program_listing_file_include_functions_FastIVM.h:

Program Listing for File FastIVM.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_functions_FastIVM.h>` (``include/functions/FastIVM.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef FAST_IVM_H
   #define FAST_IVM_H
   
   #include <mutex>
   #include <vector>
   #include <functional>
   #include <math.h>
   #include <cassert>
   #include <numeric>
   
   #include "DataTypeHandling.h"
   #include "SubmodularFunction.h"
   #include "functions/IVM.h"
   
   class FastIVM : public IVM {
   private:
       
   protected:
       // Number of items added so far. Required to maintain consistent access to kmat and L
       unsigned int added;
   
       // The kernel matrix \Sigma. 
       Matrix kmat;
   
       // The lower triangle matrix of the cholesky decomposition. Note that it stores K x K elements, even though only 1/2 * K * K + K are required for a lower triangle matrix 
       Matrix L;
   
       // The current function value
       data_t fval;
   
   public:
   
       FastIVM(unsigned int K, Kernel const &kernel, data_t sigma) : IVM(kernel, sigma), kmat(K+1), L(K+1) {
           added = 0;
           fval = 0;
       }
   
       FastIVM(unsigned int K, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
           : IVM(kernel, sigma), kmat(K+1), L(K+1) {
           added = 0;
           fval = 0;
       }
   
       data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
           if (pos >= added) {
               // Peek function value for last line
   
               for (unsigned int i = 0; i < added; ++i) {
                   data_t kval = kernel->operator()(cur_solution[i], x);
   
                   kmat(i, added) = kval;
                   kmat(added, i) = kval;
               }
               data_t kval = kernel->operator()(x, x);
               kmat(added, added) = sigma * 1.0 + kval;
   
               for (size_t j = 0; j <= added; j++) {
                   //data_t s = std::inner_product(&L[added * K], &L[added * K] + j, &L[j * K], static_cast<data_t>(0));
                   data_t s = std::inner_product(&L(added, 0), &L(added, j), &L(j,0), static_cast<data_t>(0));
                   if (added == j) {
                       L(added, j) = std::sqrt(kmat(added, j) - s);
                   } else {
                       L(added, j) = (1.0f / L(j, j) * (kmat(added, j) - s));
                   }
                   L(j, added) = L(added, j); // Symmetric update
               }
               return fval + 2.0 * std::log(L(added, added));
           } else {
               Matrix tmp(kmat, added);
               for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                   if (i == pos) {
                       data_t kval = kernel->operator()(x, x);
                       tmp(pos, pos) = sigma * 1.0 + kval;
                   } else {
                       data_t kval = kernel->operator()(cur_solution[i], x);
                       tmp(i, pos) = kval;
                       tmp(pos, i) = kval;
                   }
               }
   
               return log_det(tmp);
           }
       }
   
       void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
           if (pos >= added) {
               // TODO We often have the peek () -> update() pattern. This call can be optimized since we now basically peek twice
               fval = peek(cur_solution, x, pos);
               added++;
           } else {
               for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                   if (i == pos) {
                       data_t kval = kernel->operator()(x, x);
                       kmat(pos, pos) = sigma * 1.0 + kval;
                   } else {
                       data_t kval = kernel->operator()(cur_solution[i], x);
                       kmat(i, pos) = kval;
                       kmat(pos, i) = kval;
                   }
               }
               L = cholesky(kmat, added);
               fval = log_det_from_cholesky(L);
           }
   
       }
   
       data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
           return fval;
       }
   
       std::shared_ptr<SubmodularFunction> clone() const override {
           // We want to store k elements. To allow for efficient peeking we will reserve space for K + 1 elements in kmat and L. 
           // Thus we need to call the constructor with one element less
           return std::make_shared<FastIVM>(kmat.size() - 1, *kernel, sigma);
       }
   };
   
   #endif // FAST_IVM_H
   
