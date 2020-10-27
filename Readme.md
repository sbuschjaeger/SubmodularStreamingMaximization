# Submodular Streaming Maximization

This repository offers code for submodular function maximization with a cardinality constraint in a streaming setting. For reference there are also some batch algorithms implemented. The code focuses on easy extensiability and accasibilty. The bulk if the code is header-only C++ with a Python interface via pybind11. 

Currently supported algorithms are:

- Greedy
- SieveStreaming
- SieveStreaming++
- ThreeSieves 
- Random


We currently offer a standard implementation 

## Requirments
For building the code you need:

- CMake >= 3.13
- C++ 17 compiler (e.g. gcc-7, clang-5)

## How to use the Python interface
Probability the easiest is to use the Python interface. We use PyBind11 to generate the C++ interface which is a submodule of this repository. First clone this repo recurisvley

    git clone --recurse-submodules git@github.com:sbuschjaeger/SubmodularStreamingMaximization.git 

If you use anaconda to manage your python packages you can use the following commands to install all dependencies including the python interface PySSM 
    
    source /opt/anaconda3/bin/activate 
    conda env create -f environment.yml  --force
    conda activate pyssm

alternativly you can just install this package via

    pip install -e .

Once you installed, you can simply import the desired submodular function and optimizer. For a detailed explanation on specific parameters / functions provided please have a look at the documentation of the source code. In general each submodulare optimizer expectes the number of items to select ("K") as well as the function to optimize (e.g. the ivm objective) and each optimizers offers: 

- `fit(X)`: To fit the entire dataset X  (batch processing)
- `next(x)`: To process one element x  (stream processing)
- `get_solution`: To retrieve the current solution
- `get_fval`: To retrieve the solution's corresponding function value

A complete example which uses the Greedy algorithms to maximizse the IVM objective with RBF kernel is: 
    
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
    greedy = Greedy(K, ivm)
    
    greedy.fit(X)

    # Alternativley, you can use the streaming interface. 
    #for x in X:
    #    opt.next(x)

    fval = opt.get_fval()
    solution = np.array(opt.get_solution())

    print("Found a solution with fval = {}".format(fval))
    print(solution)

### How to provide your own submodular function
The code lies special emphasize on extending the existing code and providing your own submodular functions. 
Most commonly, people might want to use the IVM function with their own kernel. For example, consider you want to use the linear kernel then you just have to provide a function accepts two examples and pass this function to the IVM:

    import numpy as np
    from PySSM import IVM

    def linear(x1, x2):
        return np.dot(x1,x2)
    
    ivm = IVM(kernel = linear, sigma = 1.0)
    
There are two ways to provide your own submodular function. The simplest approach is to offer a python function which accepts the current set and returns the corresponding function value. For example, the IVMs logdet would be: 

    from PySSM import Greedy

    def logdet(X):
        # Note, that X is a list of list and not yet a numpy array
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
    greedy = Greedy(K, logdet)
    greedy.fit(X)

    # Alternativley, you can use the streaming interface. 
    #for x in X:
    #    opt.next(x)

    fval = opt.get_fval()
    solution = np.array(opt.get_solution())

    print("Found a solution with fval = {}".format(fval))
    print(solution)

This calls `logdet` for each function evaluation, which means that we evaluate `logdet` for every item in the dataset / stream.

### How to provide your own submodular function class

In the previous exampke we re-compute the entire kernel matrix each time we call `logdet` which becomes very costly. You can implement your own `SubmodularFunction` to cache the kernel matrix inbetween computations. To do so you need to implement the abstract SubmodularFunction class which requires three methods: 

- `peek(self, X, x)`: We peek what the current function value would be, if we _would_ add x to the current solution X. This function gets called for every item in the batch / stream so it sould be as fast as possible
- `update(self, X, x)`: We update what the current function value _is_ when x is added to the current solution X. This function gets called for every item in the batch / stream we add to our target solution.
- `clone(self)`: Some algorithms like SieveStreaming or SieveStreaming++ maintain multiple solutions with each their own SubmodularFunctions. The C++ backend will call this method method whenever it requires a new SubmodularFunction for these algorithms. 

A complete example which caches the kernel matrix inbetween computations is then:    
    
    from PySSM import SubmodularFunction

    # We implement a SubmodularFunction
    class FastLogdet(SubmodularFunction):
        def __init__(self, K):
            super().__init__()
            self.added = 0
            self.K = K
            self.kmat = np.zeros((K,K))
        
        # We peek what the current function value would be, if we would add x to the current solution X. 
        # This function gets called for every item in the batch / stream so it sould be as fast as possible
        def peek(self, X, x):
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

        # We update what the current function value is when x is added to the current solution X. 
        # This function gets called for every item in the batch / stream we add to our target solution.
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

        # Some algorithms like SieveStreaming or SieveStreaming++ maintain multiple solutions with each their own SubmodularFunctions. The C++ backend will call the clone() method whenever it requires a new SubmodularFunction for these algorithms. 
        def clone(self):
            return FastLogdet(self.K)

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
    fastIVM = FastLogdet(K)
    greedy = Greedy(K, fastIVM)
    
    greedy.fit(X)

    # Alternativley, you can use the streaming interface. 
    #for x in X:
    #    opt.next(x)

    fval = opt.get_fval()
    solution = np.array(opt.get_solution())

    print("Found a solution with fval = {}".format(fval))
    print(solution)

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

### How to provide your own optimizer
Currently we do not support writing your own optimizer in Python and we do not have any plans to do so. If you are interested in this just leave an issue or write us an E-Mail (sebastian.buschjaeger@tu-dortmund.de). Alternativley, you can implement your own maximization algorithm in C++ as explained in the next section.

## How to use the C++ interface

The c++ interface generally follows the Python interface, but offers a little bit more freedom. The code uses a generic `data_t` datatype which is defined in "DataTypeHandling.h". This is per default a `double`. As above, the optimizers will generally offer:

- `fit(std::vector<std::vector<data_t>> const &X)`: To fit the entire dataset X. 
- `next(std::vector<data_t> const &x)`: To process one element x (stream processing).The data type `data_t` is defined in "DataTypeHandling.h" and probabilty set to double (default)
- `std::vector<std::vector<data_t>>const &  get_solution()`: To retrieve a reference to the current solution. 
- `data_t get_fval()`: To retrieve the solution's corresponding function value. The data type `data_t` is defined in "DataTypeHandling.h" and probabilty set to double (default)

### How to provide your own submodular function

### How to provide your own submodular function class

### How to provide your own optimizer