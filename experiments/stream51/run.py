#!/usr/bin/env python3

import os
from scipy.io import arff
import numpy as np
import pandas as pd

import numpy as np
from numpy.linalg import slogdet
import time

from joblib import Parallel, delayed
from sklearn import preprocessing

# from PySSM import Matrix, Vector
from PySSM import RBFKernel
from PySSM import IVM, FastIVM

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 
from PySSM import Salsa 

def eval(options, X):
    name = options["method"]
    sigma = options["sigma"]
    scale = options["scale"]
    K = options["K"]
    
    reps = options.get("reps",1)
    fvals = []
    runtimes = []
    for i in range(reps):
        kernel = RBFKernel(sigma=sigma,scale=scale)
        fastLogDet = FastIVM(K, kernel, 1.0)

        nice_name = ""
        if name == "Greedy":
            opt = Greedy(K, fastLogDet)
            nice_name = name + "_" + str(K)
        elif name == "Random":
            opt = Random(K, fastLogDet, i)
            nice_name = name + "_" + str(K)
        elif name == "SieveStreaming":
            e = options["epsilon"]
            opt = SieveStreaming(K, fastLogDet, 1.0, e)
            nice_name = name + "_" + str(K) + "_" + str(e)
        elif name == "SieveStreaming++":
            e = options["epsilon"]
            opt = SieveStreamingPP(K, fastLogDet, 1.0, e)
            nice_name = name + "_" + str(K) + "_" + str(e)
        elif name == "Salsa":
            e = options["epsilon"]
            opt = Salsa(K, fastLogDet, 1.0, e)
            nice_name = name + "_" + str(K) + "_" + str(e)
        elif name == "ThreeSieves":
            e = options["epsilon"]
            T = options["T"]
            nice_name = name + "_" + str(K) + "_" + str(e) + "_" + str(T)
            opt = ThreeSieves(K, fastLogDet, 1.0, e, "sieve", T)
        
        start = time.process_time()
        if name in ["Greedy", "Salsa"]:
            opt.fit(X)
        else:
            for x in X:
                opt.next(x)
        end = time.process_time()
        fval = opt.get_fval()
        fvals.append(fval)
        runtimes.append(end - start)

    solution = opt.get_solution()
    solution_dict = {
            **options,
        "fval":np.mean(fval),
        "runtime":np.mean(runtimes),
        "solution":solution
    }
    print("{}: {}".format(nice_name, np.mean(fval)))

    np.save("/home/buschjae/projects/SubmodularStreamingMaximization/experiments/stream51/{}.npy".format(nice_name),solution_dict,allow_pickle=True)

    return {
        **options,
        "fval":np.mean(fval),
        "runtime":np.mean(runtimes)
    }

print("Loading data")
X = np.load("/data/s1/buschjae/SubmodularStreamingMaximization/stream51.npy")

min_max_scaler = preprocessing.Normalizer()
X = min_max_scaler.fit_transform(X)

#Ks = range(5,55,5)
Ks = range(55,105,5)
# Ks = [50]
eps = [1e-3,5e-3,1e-2,5e-2,1e-1]
Ts = [500, 1000, 2500, 5000]
#Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])
Sigmas = [2] #np.sqrt(X.shape[1])

results = []
runs = []
for K in Ks:
    for s in Sigmas:
        runs.append(
            ( {   
                "method": "Greedy",
                "K":K,
                "sigma":s,
                "scale":1
            }, X)
        )

        runs.append(
            ( {   
                "method": "Random",
                "K":K,
                "sigma":s,
                "scale":1,
                "reps":5
            }, X)
        )

        for e in eps:
            runs.append(
                ( {   
                    "method": "SieveStreaming",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "reps":1,
                    "epsilon":e
                }, X)
            )

            runs.append(
                ( {   
                    "method": "SieveStreaming++",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "reps":1,
                    "epsilon":e
                }, X)
            )

            runs.append(
                ( {   
                    "method": "Salsa",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "reps":1,
                    "epsilon":e
                }, X)
            )

            for T in Ts:    
                runs.append(
                    ( {   
                        "method": "ThreeSieves",
                        "K":K,
                        "sigma":s,
                        "scale":1,
                        "reps":1,
                        "epsilon":e,
                        "T":T
                    }, X)
                )

n_cores = 5
print("Running {} on {} cores".format(len(runs), n_cores))
results = Parallel(n_jobs=n_cores)(delayed(eval)(options = options, X = X) for options, X in runs)
df = pd.DataFrame(results)
df.to_csv("results.csv",index=False)
