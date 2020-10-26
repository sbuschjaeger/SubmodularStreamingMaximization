#!/usr/bin/env python3

import numpy as np
from numpy.linalg import slogdet

from PySSM import RBFKernel
from PySSM import Greedy
from PySSM import IVM
from PySSM import SubmodularFunction

def logdet(X):
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

class FastLogdet(SubmodularFunction):
    def __init__(self, K):
        super().__init__()
        self.added = 0
        self.K = K
        self.kmat = np.zeros((K,K))
        
    def peek(self, X, x):
        if self.added == 0:
            return 0
        
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

    # def clone(self):
    #     return FastLogdet()

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
# kernel = RBFKernel(sigma=1,scale=1)
# slowIVM = IVM(kernel = kernel, sigma = 1.0)
# greedy = Greedy(K, slowIVM)

#greedy = Greedy(K, logdet)
fastLogDet = FastLogdet(K)
greedy = Greedy(K, fastLogDet)

greedy.fit(X)
fval = greedy.get_fval()
solution = np.array(greedy.get_solution())

print("Found a solution with fval = {}".format(fval))
print(solution)
