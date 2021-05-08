
.. _program_listing_file_include_functions_IVM.h:

Program Listing for File IVM.h
==============================

|exhale_lsh| :ref:`Return to documentation for file <file_include_functions_IVM.h>` (``include/functions/IVM.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef INFORMATIVE_VECTOR_MACHINE_H
   #define INFORMATIVE_VECTOR_MACHINE_H
   
   #include <mutex>
   #include <vector>
   #include <functional>
   #include <math.h>
   #include <cassert>
   #include "DataTypeHandling.h"
   #include "SubmodularFunction.h"
   #include "kernels/Kernel.h"
   #include "functions/Matrix.h"
   
   class IVM : public SubmodularFunction {
   protected:
   
       inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X, data_t sigma) const {
           unsigned int K = X.size();
           Matrix mat(K);
   
           for (unsigned int i = 0; i < K; ++i) {
               for (unsigned int j = i; j < K; ++j) {
                   data_t kval = kernel->operator()(X[i], X[j]);
                   if (i == j) {
                       mat(i,j) = sigma * 1.0 + kval;
                   } else {
                       mat(i,j) = kval;
                       mat(j,i) = kval;
                   }
               }
           }
   
           // TODO CHECK IF THIS USES MOVE
           return mat;
       }
   
       // The kernel
       std::shared_ptr<Kernel> kernel;
   
       // The scaling constant
       data_t sigma;
   
   public:
   
       IVM(Kernel const &kernel, data_t sigma) : kernel(kernel.clone()), sigma(sigma) {
           assert(("The sigma value of the IVM should be greater than  0!", sigma > 0));
       }
   
       IVM(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
           : kernel(std::unique_ptr<Kernel>(new KernelWrapper(kernel))), sigma(sigma) {
           assert(("The sigma value of the IVM should be greater than  0!", sigma > 0));
       }
   
       data_t peek(std::vector<std::vector<data_t>> const& cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
           std::vector<std::vector<data_t>> tmp(cur_solution);
   
           if (pos >= cur_solution.size()) {
               tmp.push_back(x);
           } else {
               tmp[pos] = x;
           }
   
           data_t ftmp = this->operator()(tmp);
           return ftmp;
       } 
   
       void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {}
   
       data_t operator()(std::vector<std::vector<data_t>> const &X) const override {
           // This is the most basic implementations which recomputes everything with each call
           // I would not use this for any real-world problems. 
           
           Matrix kernel_mat = compute_kernel(X);
           return log_det(kernel_mat);
       } 
   
       std::shared_ptr<SubmodularFunction> clone() const override {
           return std::make_shared<IVM>(*kernel, sigma);
       }
   
       ~IVM() {/* Nothing do to, since the shared_pointer should clean-up itself*/ }
   };
   
   #endif // INFORMATIVE_VECTOR_MACHINE_H
   
