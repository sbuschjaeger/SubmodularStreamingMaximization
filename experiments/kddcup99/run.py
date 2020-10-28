#!/usr/bin/env python3

import os
from scipy.io import arff
import numpy as np
import pandas

import numpy as np
from numpy.linalg import slogdet
import time

# from PySSM import Matrix, Vector
from PySSM import RBFKernel
from PySSM import IVM, FastIVM

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 

#from PySSM import fit_greedy_on_ivm , fit_greedy_on_ivm_2

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

def evaluate_batch(opt, X):
    start = time.process_time()
    opt.fit(X)
    fval = opt.get_fval()
    end = time.process_time()
    solution = np.array(opt.get_solution())
    return {
        "solution":solution,
        "fval":fval,
        "runtime":end - start
    }

def evaluate_stream(opt, X):
    start = time.process_time()
    for x in X:
        opt.next(x)
    fval = opt.get_fval()
    end = time.process_time()
    solution = np.array(opt.get_solution())
    return {
        "solution":solution,
        "fval":fval,
        "runtime":end - start
    }

print("Loading data")
data, meta = arff.loadarff(os.path.join(os.path.dirname(__file__), "data", "KDDCup99", "KDDCup99_withoutdupl_norm_1ofn.arff"))

data_pd = pandas.DataFrame(data)
data_pd.columns = meta

# Extract label vector
y = np.array([-1 if x == "yes" else 1 for x in data_pd["outlier"]])  # 1 = inlier, -1 = outlier

# Delete irrelevant features.
data_pd = data_pd.drop("outlier", axis=1)
data_pd = data_pd.drop("id", axis=1)

# Only values from now on
X = data_pd.values

K = 20

kernel = RBFKernel(sigma=np.sqrt(X.shape[1]),scale=1)
fastLogDet = FastIVM(K, kernel, 1.0)

# TODO 
# xmat = Matrix()
# for x in X:
#     xvec = Vector(x)
#     xmat.append(xvec)

# print("Selecting {} represantatives via all c++ greedy".format(K))
# start = time.process_time()
# fval = fit_greedy_on_ivm(K=K, sigma=np.sqrt(X.shape[1]), scale=1.0, epsilon = 1.0, X = xmat)
# end = time.process_time()
# runtime = end - start
# print("\t fval:\t{} \n\t runtime:\t{} \n\n".format(fval, runtime))

# start = time.process_time()
# fval = fit_greedy_on_ivm_2(K=K, sigma=1.0, scale=2.0, epsilon = 3.0)
# end = time.process_time() 
# runtime = end - start
# print("\t fval:\t{} \n\t runtime:\t{} \n\n".format(fval, runtime))
print()
print("=== BATCH PROCESSING ===")
print()

print("Selecting {} represantatives via Greedy with C++ logdet".format(K))
res = evaluate_batch(Greedy(K, fastLogDet), X)
print("\t fval:\t{} \n\t runtime:\t{} \n\n".format(res["fval"], res["runtime"]))

# print("Selecting {} represantatives via Greedy with python logdet".format(K))
# res = evaluate_optimizer(Greedy(K, logdet), X)
# print("\t fval:\t{} \n\t runtime:\t{} \n\n".format(res["fval"], res["runtime"]))

print("Selecting {} represantatives via Random".format(K))
res = evaluate_batch(Random(K, fastLogDet), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

print("Selecting {} represantatives via Sieve".format(K))
res = evaluate_batch(SieveStreaming(K, fastLogDet, 1.0, 0.01), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

# print("Selecting {} represantatives via Sieve++".format(K))
# res = evaluate_batch(SieveStreamingPP(K, fastLogDet, 1.0, 0.01), X)
# print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

print("Selecting {} represantatives via ThreeSieves".format(K))
res = evaluate_batch(ThreeSieves(K, fastLogDet, 1.0, 0.01, "sieve", 1000), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

print()
print("=== STREAM PROCESSING ===")
print()

print("Selecting {} represantatives via Random".format(K))
res = evaluate_stream(Random(K, fastLogDet), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

print("Selecting {} represantatives via Sieve".format(K))
res = evaluate_stream(SieveStreaming(K, fastLogDet, 1.0, 0.01), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

# print("Selecting {} represantatives via Sieve++".format(K))
# res = evaluate_stream(SieveStreamingPP(K, fastLogDet, 1.0, 0.01), X)
# print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

print("Selecting {} represantatives via ThreeSieves".format(K))
res = evaluate_stream(ThreeSieves(K, fastLogDet, 1.0, 0.01, "sieve", 1000), X)
print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))
