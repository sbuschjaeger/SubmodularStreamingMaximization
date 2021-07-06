Using the Python Interface
==========================

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

Implementing a custom kernel for the IVM
----------------------------------------

Implementing your own custom kernel is easy. To do so, there are two options. Either you simply implement a python function accepting two arguments `x1` and `x2` and simply pass them to the IVM / FastIVM object. We recommend this approach for stateless kernel functions.
Alternatively, you can extend the `Kernel` class by implementing the `clone` and `__call__` method. The `clone` method must return a clone (_not_ a copy) of the object, whereas the `__call__` method accepts two the two instances `x1` and `x2`. Use this approach if you want to implement a stateful kernel. 

*Note*: The parameters `x1` and `x2` are regular python lists. Make sure to transform them to the appropriate data types before using them.

.. code-block:: python

   from PySSM import Kernel
   
   # Define a new kernel by implementing the clone and __call__ function
   class PolyKernel(Kernel): 
    def clone(self):
        return PolyKernel()

    def __call__(self,x1,x2):
        return np.dot(np.array(x1), np.array(x2))
   
   # Define a new kernel by implementing the function directly
   def poly_kernel(x1,x2):
      return np.dot(np.array(x1), np.array(x2))

   # Use the class interface
   kernel = PolyKernel()
   ivm = FastIVM(K, kernel = kernel, sigma = 1.0)
   
   #Alternatively use the function directly
   #ivm = FastIVM(K, kernel = poly_kernel, sigma = 1.0)

Implementing custom submodular functions
----------------------------------------

Implementing your own submodular function is easy. Again there are two options: First, you simply provide a python function which evaluates the function value of the provided summary `X`. Any optimizer accepts these regular python functions and an example is given below which computes the kernel matrix of the provided summary and computes its logdet via numpys `slogdet` method. We recommend this approach if you want to implement stateless submodular functions.

Re-computing the kernel matrix can become slow for larger summaries. Thus, you can also implement the SubmodularFunction interface directly to cache computations. To do so, you have to implement the `peek`, the `update`, the `clone` and the `__call__`  method. For more details please see the dedicated documentation for the C++ back-end of SubmodularFunction. An example is given below. 

*Note*: The parameters `X` and `x` are regular python lists. Make sure to transform them to the appropriate data types before using them.


.. code-block:: python

   from numpy.linalg import slogdet
   from PySSM import SubmodularFunction

   # Compute the kernel matrix + its logdet
   def ivm(X):
      X = np.array(X)
      K = X.shape[0]
      kmat = np.zeros((K,K))

      for i, xi in enumerate(X):
         for j, xj in enumerate(X):
               kval = 1.0*np.exp(-np.sum((xi-xj)**2) / 1.0)
               if i == j:
                  kmat[i][i] = 1.0 + kval / 1.0**2
               else:
                  kmat[i][j] = kval / 1.0**2
                  kmat[j][i] = kval / 1.0**2
      return slogdet(kmat)[1]

   # This is a dummy implementation of the IVM function which caches the kernel matrix
   class FastLogdet(SubmodularFunction):
      def __init__(self, K):
         super().__init__()
         self.added = 0
         self.K = K
         self.kmat = np.zeros((K,K))
         
      def peek(self, X, x):
         # if self.added == 0:
         #     return 0
         
         if self.added < self.K:
               #X = np.array(X)
               x = np.array(x)
               
               row = []
               for xi in X:
                  kval = 1.0*np.exp(-np.sum((xi-x)**2) / 1.0)
                  row.append(kval)
               kval = 1.0*np.exp(-np.sum((x-x)**2) / 1.0)
               row.append(1.0 + kval / 1.0**2)

               self.kmat[:self.added, self.added] = row[:-1]
               self.kmat[self.added, :self.added + 1] = row
               return slogdet(self.kmat[:self.added + 1,:self.added + 1])[1]
         else:
               return 0

      def update(self, X, x):
         #X = np.array(X)
         if self.added < self.K:
               x = np.array(x)

               row = []
               for xi in X:
                  kval = 1.0*np.exp(-np.sum((xi-x)**2) / 1.0)
                  row.append(kval)

               kval = 1.0*np.exp(-np.sum((x-x)**2) / 1.0)
               row.append(1.0 + kval / 1.0**2)

               self.kmat[:self.added, self.added] = row[:-1]
               self.kmat[self.added, :self.added + 1] = row
               self.added += 1

               return slogdet(self.kmat[:self.added + 1,:self.added + 1])[1]
         else:
               return 0

      def clone(self):
         return FastLogdet(self.K)

      def __call__(self, X):
         return ivm(X)

Implementing your own optimizer
-------------------------------

We currently do not support the implementation of new optimizers in Python, but only in C++. Sorry. Please have a look at the C++ documentation. 