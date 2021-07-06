
.. _program_listing_file_include_SubmodularFunction.h:

Program Listing for File SubmodularFunction.h
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_SubmodularFunction.h>` (``include/SubmodularFunction.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef SUBMODULARFUNCTION_H
   #define SUBMODULARFUNCTION_H
   
   #include <memory>
   #include <stdexcept>
   #include <utility>
   #include <vector>
   #include <functional>
   #include <cassert>
   
   #include "DataTypeHandling.h"
   
   
   class SubmodularFunction {
   public:
       // TODO THIS SHOULD NOT BE NEEDED. WHY DO WE HAVE THIS?!?!
       virtual data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const = 0;
   
       virtual data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) = 0; 
   
       virtual void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) = 0;
   
       virtual std::shared_ptr<SubmodularFunction> clone() const = 0;
   
       virtual ~SubmodularFunction() {}
   };
   
   class SubmodularFunctionWrapper : public SubmodularFunction {
   protected:
       // The std::function which implements the actual submodular function
       std::function<data_t (std::vector<std::vector<data_t>> const &)> f;
   
   public:
   
       SubmodularFunctionWrapper(std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : f(f) {}
   
       data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const {
           return f(cur_solution);
       }
   
       data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) {
           std::vector<std::vector<data_t>> tmp(cur_solution);
   
           if (pos >= cur_solution.size()) {
               tmp.push_back(x);
           } else {
               tmp[pos] = x;
           }
   
           data_t ftmp = this->operator()(tmp);
           return ftmp;
       }
   
       void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) {}
       
       std::shared_ptr<SubmodularFunction> clone() const {
           return std::shared_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f));
       }
   
       ~SubmodularFunctionWrapper() {}
   };
   
   #endif // SUBMODULARFUNCTION_H
