Using the C++ interface
=======================

The C++ code is header-only so simply include the desired functions in your project and your are good to go. If you have trouble compiling you can look at the ``CMakeLists.txt`` file which compiles the Python bindings as well as the test files For a detailed explanation on specific parameters / functions provided please have a look at the documentation of the individual optimizers. Generally, each optimizer provides

- ``fit(X)``: Selects a summary of the given data set (batch processing)
- ``next(x)``: Consumes the next data item from a stream  (stream processing)
- ``get_solution()``: Returns the current solution 
- ``get_ids()``: Returns the id (if any) of each object
- ``get_num_candidate_solutions``: Returns the number of intermediate solutions stored by the optimizer
- ``get_num_elements_stored``: Returns the number of elements stored by the optimizer
- ``get_fval``: Returns the function value

. The following example uses the Greedy optimizer to select a data summary by maximizing the Informative Vector Machine (the full examples can be found in ``tests/main.cpp``\ )

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include <math.h>
   #include "FastIVM.h"
   #include "RBFKernel.h"
   #include "Greedy.h"

   std::vector<std::vector<double>> data = {
         {0, 0},
         {1, 1},
         {0.5, 1.0},
         {1.0, 0.5},
         {0, 0.5},
         {0.5, 1},
         {0.0, 1.0},
         {1.0, 0.0}
   };    

   unsigned int K = 3;
   FastIVM fastIVM(K, RBFKernel(), 1.0);

   Greedy greedy(K, fastIVM)
   greedy.fit(data);
   auto solution = greedy.get_solution();
   double fval = greedy.get_fval();

   std::cout << "Found a solution with fval = " << fval << std::endl;
   for (auto x : solution) {
         for (auto xi : x) {
            std::cout << xi << " ";
         }
         std::cout << std::endl;
   }

Implementing a custom kernel for the IVM
----------------------------------------

Implementing your own custom kernel is easy. To do so, there are two options. Either you simply implement a regular function accepting two arguments `x1` and `x2` and  pass them to the IVM / FastIVM object. We recommend this approach for stateless kernel functions.
Alternatively, you can extend the `Kernel` class by implementing the `clone` method and the `operator()` method. The `clone` method must return a `shared_ptr` to a clone (__not__ a copy) of the object, whereas the `operator()` method accepts the two instances `x1` and `x2`. Use this approach if you want to implement a stateful kernel. 

.. code-block:: cpp

   data_t poly_kernel(const std::vector<data_t>& x1, const std::vector<data_t>& x2) {
      data_t distance = 0;
      if (x1 != x2) {
         for (unsigned int i = 0; i < x1.size(); ++i) {
               distance += x1[i]*x2[i];
         }
      }
      return distance;
   }


   class PolyKernel : public Kernel {
   public:
      PolyKernel() = default;

      inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
          data_t distance = 0;
         if (x1 != x2) {
            for (unsigned int i = 0; i < x1.size(); ++i) {
                  distance += x1[i]*x2[i];
            }
         }
         return distance;
      }

      std::shared_ptr<Kernel> clone() const override {
         return std::shared_ptr<Kernel>(new PolyKernel());
      }
   };

Implementing custom submodular functions
----------------------------------------

Implementing your own submodular function is easy. Again there are two options: First, you simply provide a regular function which evaluates the function value of the provided summary `X`. Any optimizer accepts these regular functions and an example is given below which computes the logdet of the kernel matrix via the Matrix class. We recommend this approach if you want to implement stateless submodular functions.

Re-computing the kernel matrix can become slow for larger summaries. Thus, you can also implement the SubmodularFunction interface directly to cache computations. To do so, you have to implement the `peek`, the `update`, the `clone` and the `operator()`  method. For more details please see the dedicated documentation for SubmodularFunction. An example is given below. 

.. code-block:: cpp

   inline data_t logdet(std::vector<std::vector<data_t>> const &cur_solution) {
      unsigned int K = X.size();
      Matrix kmat(K);

      for (unsigned int i = 0; i < K; ++i) {
         for (unsigned int j = i; j < K; ++j) {
               data_t kval = poly_kernel(X[i], X[j]);
               if (i == j) {
                  kmat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
               } else {
                  kmat(i,j) = kval / std::pow(1.0, 2.0);
                  kmat(j,i) = kval / std::pow(1.0, 2.0);
               }
         }
      }
      return log_det(kmat, cur_solution.size());
   }


   class FastLogDet : public SubmodularFunction {
   private:
      
   protected:
      // Number of items added so far. Required to maintain consistent access to kmat and L
      unsigned int added;

      // The kernel matrix \Sigma. 
      // See Matrix.h for more details
      Matrix kmat;

   public:

      FastIVM(unsigned int K) : kmat(K+1) {
         added = 0;
      }


      data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
         if (pos >= added) {
               // Peek function value for last line

               for (unsigned int i = 0; i < added; ++i) {
                  data_t kval = poly_kernel(cur_solution[i], x);

                  kmat(i, added) = kval;
                  kmat(added, i) = kval;
               }
               data_t kval = poly_kernel(x, x);
               kmat(added, added) = 1.0 + kval;

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
                     data_t kval = poly_kernel(x, x);
                     tmp(pos, pos) = 1.0 + kval;
                  } else {
                     data_t kval = poly_kernel(cur_solution[i], x);
                     tmp(i, pos) = kval;
                     tmp(pos, i) = kval;
                  }
               }

               return log_det(tmp);
         }
      }

      void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
         if (pos >= added) {
               fval = peek(cur_solution, x, pos);
               added++;
         } else {
               for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                  if (i == pos) {
                     data_t kval = poly_kernel(x, x);
                     kmat(pos, pos) = 1.0 + kval;
                  } else {
                     data_t kval = poly_kernel(cur_solution[i], x);
                     kmat(i, pos) = kval;
                     kmat(pos, i) = kval;
                  }
               }
               L = cholesky(kmat, added);
               fval = log_det_from_cholesky(L);
         }

      }

      data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
         return log_det(kmat);
      }

      std::shared_ptr<SubmodularFunction> clone() const override {
         // We want to store k elements. To allow for efficient peeking we will reserve space for K + 1 elements in kmat and L. 
         // Thus we need to call the constructor with one element less
         return std::make_shared<FastIVM>(kmat.size() - 1);
      }
   };

Implementing your own optimizer
-------------------------------

Implementing your own optimizer is more challenging and requires some background in Python and C++. To do so, you first must implement the SubmodularOptimizer interface which requires you to implement the `next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt)` method. The `next` method consumes the next item in the data stream and -- depending on the method -- adds it to the summary or not. An optional identifier is also supplied which might be used to uniquely identify items  from the stream. Make sure to correctly call `update` and `peek`/`operator` of the SubmodularFunction to store the correct function values. In addition, you must provide two constructors for your optimizer which both accept the number of elements to select K as well as the submodular function (either as `std::function` or as SubmodularFunction object). For more information please consult the documentation of the SubmodularOptimizer interface. As a simple example consider the following random sampling algorithm which uses reservoir sampling:

.. code-block:: cpp

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


If you only want to use the C++ interface, then you are already done after implementing your class. If you also want to expose the implementation to Python then you will need to implement additional PyBind bindings. To do so, you need to add your bindings to `Python.cpp`. Please consult the PyBind documentation if you are not familiar with PyBind. In most cases however the pattern usually looks something like this:

.. code-block:: cpp

   py::class_<MyNewOptimizer>(m, "MyNewOptimizer") 
        /* These are the constructor definitions for your optimizer which probably include some additional options. Make sure that the data-types match. */
        .def(py::init<unsigned int, SubmodularFunction&, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("option1")= 0)
        .def(py::init<unsigned int, std::function<data_t (std::vector<std::vector<data_t>> const &)>, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("option1") = 0)
        /* These functions are likely unchanged */
        .def("get_solution", &MyNewOptimizer::get_solution)
        .def("get_ids", &MyNewOptimizer::get_ids)
        .def("get_fval", &MyNewOptimizer::get_fval)
        .def("get_num_candidate_solutions", &MyNewOptimizer::get_num_candidate_solutions)
        .def("get_num_elements_stored", &MyNewOptimizer::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<std::vector<data_t>> const &, unsigned int>(&MyNewOptimizer::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<std::vector<data_t>> const &, std::vector<idx_t> const &, unsigned int>(&MyNewOptimizer::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &MyNewOptimizer::next, py::arg("x"), py::arg("id") = std::nullopt);

Note that we use the `clone` function of SubmodularFunction in the constructor to make sure that stateful functions do not have side-effects. Moreover, we try to stick to "modern" C++ utilizing the stl when possible / helpful. Last, you are free to override more functions from the SubmodularOptimizer interface and/or expose more functions to the python-side. For example, we may also override the `fit` function in the above Random example to directly sample data-points instead of using reservoir sampling.