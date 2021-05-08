Welcome to SubmodularStreamMaximization's documentation!
=========================================================

.. toctree::
   :hidden:
   :maxdepth: 2
   
   api/library_root

This repository contains code for Submodular Streaming Maximization. The code focuses on easy extensibility and accessibility. It is mainly written in header-only C++ with a Python interface via pybind11. Currently implemented algorithms are:

* :class:`Greedy`
* :class:`SieveStreaming`
* :class:`SieveStreaming++`
* :class:`ThreeSieves`
* :class:`Random`
* :class:`Salsa`
* :class:`IndependentSetImprovement`

Supported submodular functions are:

* Informative Vector Machine (:class:`IVM`) (sometimes called LogDet) with a custom kernel. Look at this function if you want to implement your own submodular function.
* Fast Informative Vector Machine (:class:`FastIVM`) with a custom kernel. This keeps track of the Cholesky Decomposition of the kernel matrix and updates it without re-computing the entire Kernelmatrix or its inverse. Use this function if speed is important. 
* :class:`RBFKernel` The RBF kernel for both IVM variants. Look at this code if you want to implement your own kernel.

For building the code you need:

* CMake >= 3.13
* C++ 17 compiler (e.g. gcc-7, clang-5)


Using the Python Interface
--------------------------

If you use anaconda to manage your python packages you can use the following commands to install all dependencies including the python interface ``PySSM`` 

.. code-block::

   conda env create -f environment.yml  --force
   conda activate pyssm


Note the ``--force`` option which overrides all existing environments called ``pyssm``. Alternatively, you can just install this package via

.. code-block::

   pip install -e .


Once installed, you can simply import the desired submodular function and optimizer via ``PySSM``. For a detailed explanation on specific parameters / functions provided please have a look at the documentation of the individual optimizers. Generally, each optimizer provides

- ``fit(X)``: Selects a summary of the given data set (batch processing)
- ``next(x)``: Consumes the next data item from a stream  (stream processing)
- ``get_solution()``: Returns the current solution 
- ``get_ids()``: Returns the id (if any) of each object
- ``get_num_candidate_solutions``: Returns the number of intermediate solutions stored by the optimizer
- ``get_num_elements_stored``: Returns the number of elements stored by the optimizer
- ``get_fval``: Returns the function value

The following example uses the Greedy optimizer to select a data summary by maximizing the Informative Vector Machine (the full examples can be found in ``tests/main.py``\ )

.. code-block:: python

   from PySSM import RBFKernel
   from PySSM import IVM
   from PySSM import Greedy

   X = [
       [0, 0],
       [1, 1],
       [0.5, 1.0],
       [1.0, 0.5],
       [0, 0.5],
       [0.5, 1],
       [0.0, 1.0],
       [1.0, 0.]
   ]    

   K = 3
   kernel = RBFKernel(sigma=1,scale=1)
   ivm = IVM(kernel = kernel, sigma = 1.0)
   opt = Greedy(K, ivm)

   opt.fit(X)

   # Alternativley, you can use the streaming interface. 
   #for x in X:
   #    opt.next(x)

   fval = opt.get_fval()
   solution = np.array(opt.get_solution())

   print("Found a solution with fval = {}".format(fval))
   print(solution)

Using the C++ interface
-----------------------

The C++ code is header-only so simply include the desired functions in your project and your are good to go. If you have trouble compiling you can look at the ``CMakeLists.txt`` file which compiles the Python bindings as well as the following test file. The following example uses the Greedy optimizer to select a data summary by maximizing the Informative Vector Machine (the full examples can be found in ``tests/main.cpp``\ )

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`