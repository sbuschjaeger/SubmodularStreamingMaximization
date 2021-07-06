#!/usr/bin/env python3

import numpy as np
import sys
from numpy.linalg import slogdet

from PySSM import Kernel
from PySSM import RBFKernel
from PySSM import IVM, FastIVM
from PySSM import SubmodularFunction

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 
from PySSM import IndependentSetImprovement
from PySSM import Salsa

# Polynomial kernel / linear kernel implemented as a class
class PolyKernel(Kernel): 
    def clone(self):
        return PolyKernel()

    def __call__(self,x1,x2):
        return np.dot(np.array(x1), np.array(x2))/len(x1)
    
# Polynomial kernel / linear kernel implemented as a function
def poly_kernel(x1,x2):
    return np.dot(np.array(x1), np.array(x2))/len(x1)

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
        self.kmat = np.zeros((K+1,K+1))
        
    def peek(self, X, x, pos):
        if pos >= self.added:
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
            x = np.array(x)
            
            row = []
            for xi in X:
                kval = 1.0*np.exp(-np.sum((xi-x)**2) / 1.0)
                row.append(kval)
            kval = 1.0*np.exp(-np.sum((x-x)**2) / 1.0)
            row.append(1.0 + kval / 1.0**2)

            row_old = self.kmat[:pos, pos]
            self.kmat[:pos, pos] = row[:-1]
            self.kmat[pos, :pos + 1] = row
            tmp = slogdet(self.kmat[:self.added + 1,:self.added + 1])[1]
            
            self.kmat[:pos, pos] = row_old[:-1]
            self.kmat[pos, :pos + 1] = row_old

            return tmp

    def update(self, X, x, pos):
        if pos >= self.added:
            self.peek(X, x, pos)
            self.added += 1
        else:
            x = np.array(x)

            row = []
            for xi in X:
                kval = 1.0*np.exp(-np.sum((xi-x)**2) / 1.0)
                row.append(kval)

            kval = 1.0*np.exp(-np.sum((x-x)**2) / 1.0)
            row.append(1.0 + kval / 1.0**2)

            self.kmat[:pos, pos] = row[:-1]
            self.kmat[pos, :pos + 1] = row

            return slogdet(self.kmat[:self.added + 1,:self.added + 1])[1]

    def clone(self):
        return FastLogdet(self.kmat.shape[0] - 1)

    def __call__(self, X):
        return ivm(X)

# Generate some dummy data with K = 3 different points
X = [
    [0.0,0.0],
    [1.0,1.0],
    [0.0,1.0],
    [0.0,0.0],
    [1.0,1.0],
    [0.0,1.0],
    [0.0,0.0],
    [1.0,1.0],
    [0.0,1.0],
    [0.0,0.0],
    [1.0,1.0],
    [0.0,1.0], 
]
K = 3

# Target solutions depending on the kernel
target_rbf = np.array(sorted([
    [0.0,0.0],
    [1.0,1.0],
    [0.0,1.0]
]))

target_poly = np.array(sorted([
    [0.0,1.0],
    [1.0,1.0], 
    [1.0,1.0]
]))

# Define all the kernel / submodular function combinations
kernel = RBFKernel(sigma=1,scale=1)
ivm_rbf = FastIVM(K, kernel = kernel, sigma = 1.0)

kernel = PolyKernel()
ivm_custom_kernel_class = FastIVM(K, kernel = kernel, sigma = 1.0)
ivm_custom_kernel_function = FastIVM(K, kernel = poly_kernel, sigma = 1.0)

ivm_custom_class = FastLogdet(K)
ivm_custom_function = ivm

optimizers = {}

### GREEDY ### 
optimizers["Greedy with IVM + RBF"] = Greedy(K, ivm_rbf)
optimizers["Greedy with IVM + poly kernel class"] = Greedy(K, ivm_custom_kernel_class)
optimizers["Greedy with IVM + poly kernel function"] = Greedy(K, ivm_custom_kernel_function)
optimizers["Greedy with custom IVM class"] = Greedy(K, ivm_custom_class)
optimizers["Greedy with custom IVM function"] = Greedy(K, ivm_custom_function)

### Random ### 
# We "optimize" over the random seeds so that the solution matches the target solution and we do not need to distinguish 
# between Random and the other optimizers
optimizers["Random with IVM + RBF"] = Random(K, ivm_rbf, 12345)
optimizers["Random with IVM + poly kernel class"] = Random(K, ivm_custom_kernel_class, 22222)
optimizers["Random with IVM + poly kernel function"] = Random(K, ivm_custom_kernel_function, 22222)
optimizers["Random with custom IVM class"] = Random(K, ivm_custom_class, 12345)
optimizers["Random with custom IVM function"] = Random(K, ivm_custom_function, 12345)

### IndependentSetImprovement ### 
optimizers["IndependentSetImprovement with IVM + RBF"] = IndependentSetImprovement(K, ivm_rbf)
optimizers["IndependentSetImprovement with IVM + poly kernel class"] = IndependentSetImprovement(K, ivm_custom_kernel_class)
optimizers["IndependentSetImprovement with IVM + poly kernel function"] = IndependentSetImprovement(K, ivm_custom_kernel_function)
optimizers["IndependentSetImprovement with custom IVM class"] = IndependentSetImprovement(K, ivm_custom_class)
optimizers["IndependentSetImprovement with custom IVM function"] = IndependentSetImprovement(K, ivm_custom_function)

### SieveStreaming ### 
optimizers["SieveStreaming with IVM + RBF"] = SieveStreaming(K, ivm_rbf, 1.0, 0.1)
optimizers["SieveStreaming with IVM + poly kernel class"] = SieveStreaming(K, ivm_custom_kernel_class, 1.0, 0.5)
optimizers["SieveStreaming with IVM + poly kernel function"] = SieveStreaming(K, ivm_custom_kernel_function, 1.0, 0.5)
optimizers["SieveStreaming with custom IVM class"] = SieveStreaming(K, ivm_custom_class, 1.0, 0.1)
optimizers["SieveStreaming with custom IVM function"] = SieveStreaming(K, ivm_custom_function, 1.0, 0.1)

### SieveStreamingPP ### 
optimizers["SieveStreamingPP with IVM + RBF"] = SieveStreamingPP(K, ivm_rbf, 1.0, 0.1)
optimizers["SieveStreamingPP with IVM + poly kernel class"] = SieveStreamingPP(K, ivm_custom_kernel_class, 1.0, 0.1)
optimizers["SieveStreamingPP with IVM + poly kernel function"] = SieveStreamingPP(K, ivm_custom_kernel_function, 1.0, 0.1)
optimizers["SieveStreamingPP with custom IVM class"] = SieveStreamingPP(K, ivm_custom_class, 1.0, 0.1)
optimizers["SieveStreamingPP with custom IVM function"] = SieveStreamingPP(K, ivm_custom_function, 1.0, 0.1)

### Salsa ### 
optimizers["Salsa with IVM + RBF"] = Salsa(K, ivm_rbf, 1.0, 0.1)
optimizers["Salsa with IVM + poly kernel class"] = Salsa(K, ivm_custom_kernel_class, 1.0, 0.1)
optimizers["Salsa with IVM + poly kernel function"] = Salsa(K, ivm_custom_kernel_function, 1.0, 0.1)
optimizers["Salsa with custom IVM class"] = Salsa(K, ivm_custom_class, 1.0, 0.1)
optimizers["Salsa with custom IVM function"] = Salsa(K, ivm_custom_function, 1.0, 0.1)

### ThreeSieves ### 
optimizers["ThreeSieves with IVM + RBF"] = ThreeSieves(K, ivm_rbf, 1.0, 0.1, "sieve",5)
optimizers["ThreeSieves with IVM + poly kernel class"] = ThreeSieves(K, ivm_custom_kernel_class, 1.0, 0.01, "sieve",1)
optimizers["ThreeSieves with IVM + poly kernel function"] = ThreeSieves(K, ivm_custom_kernel_function, 1.0, 0.01, "sieve",1)
optimizers["ThreeSieves with custom IVM class"] = ThreeSieves(K, ivm_custom_class, 1.0, 0.1, "sieve",5)
optimizers["ThreeSieves with custom IVM function"] = ThreeSieves(K, ivm_custom_function, 1.0, 0.1, "sieve",5)

failed = False
for name, opt in optimizers.items():
    opt.fit(X)
    fval = opt.get_fval()
    solution = np.array(sorted(opt.get_solution()))

    print("Testing {}".format(name))
    print("\tfval is {}".format(fval))

    if "poly" in name:
        if not np.array_equal(solution, target_poly):
            failed = True
            print("\tTEST FAILED. Solution does not match target solution!")
            print("\tSolution was:")
            for s in solution:
                print("\t\t", s)
            print("\t...but target was:")
            for s in target_poly:
                print("\t\t", s)
        else:
            print("\tTEST PASSED. Solution matches target solution")
    else:
        if not np.array_equal(solution, target_rbf):
            failed = True
            print("\tTEST FAILED. Solution does not match target solution!")
            print("\tSolution was:")
            for s in solution:
                print("\t\t", s)
            print("\t...but target was:")
            for s in target_rbf:
                print("\t\t", s)
        else:
            print("\tTEST PASSED. Solution matches target solution")
    print("")

sys.exit(failed == True)