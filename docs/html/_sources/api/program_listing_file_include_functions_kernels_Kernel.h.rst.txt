
.. _program_listing_file_include_functions_kernels_Kernel.h:

Program Listing for File Kernel.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_functions_kernels_Kernel.h>` (``include/functions/kernels/Kernel.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef KERNEL_H
   #define KERNEL_H
   
   #include <cassert>
   #include "DataTypeHandling.h"
   
   class Kernel {
   
   public:
   
       virtual inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const = 0;
   
       virtual std::shared_ptr<Kernel> clone() const = 0;
   
       virtual ~Kernel() {}
   };
   
   class KernelWrapper : public Kernel {
   protected:
       std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> f;
   
   public:
   
       KernelWrapper(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> f) : f(f) {}
   
       inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
           return f(x1, x2);
       }
   
       std::shared_ptr<Kernel> clone() const override {
           return std::shared_ptr<Kernel>(new KernelWrapper(f));
       }
   
   };
   
   #endif // RBF_KERNEL_H
