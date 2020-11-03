#!/usr/bin/env python3

import os
from scipy.io import arff
import numpy as np
import pandas as pd

import numpy as np
from numpy.linalg import slogdet
import time
from joblib import Parallel, delayed

# from PySSM import Matrix, Vector
from PySSM import RBFKernel
from PySSM import IVM, FastIVM

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 
from PySSM import Salsa 

import os
import numpy as np
import scipy.io
import scipy.io
from sklearn import preprocessing


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

        if name == "Greedy":
            opt = Greedy(K, fastLogDet)
        elif name == "Random":
            opt = Random(K, fastLogDet, i)
        elif name == "SieveStreaming":
            e = options["epsilon"]
            opt = SieveStreaming(K, fastLogDet, 1.0, e)
        elif name == "SieveStreaming++":
            e = options["epsilon"]
            opt = SieveStreamingPP(K, fastLogDet, 1.0, e)
        elif name == "Salsa":
            e = options["epsilon"]
            opt = Salsa(K, fastLogDet, 1.0, e)
        elif name == "ThreeSieves":
            e = options["epsilon"]
            T = options["T"]
            opt = ThreeSieves(K, fastLogDet, 1.0, e, "sieve", T)
        
        start = time.process_time()
        opt.fit(X)
        fval = opt.get_fval()
        end = time.process_time()
        fvals.append(fval)
        runtimes.append(end - start)

    return {
        **options,
        "fval":np.mean(fval),
        "runtime":np.mean(runtimes)
    }

print("Loading data")

path = "/home/share02/datensaetze/fact_icdm_2020/crab/factExamples_no_cut_20131101_embedded_256.npy"
data = np.load(path)
idx = np.random.choice(range(0, data.shape[0]), 200000)
X = data[idx,:]

# MinMax normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

Ks = range(5,100,5)
# Ks = [5]
eps = [1e-3,5e-3,1e-2,5e-2,1e-1]
Ts = [500, 1000, 2500, 5000]
#Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])
Sigmas = [np.sqrt(X.shape[1])]

results = []
runs = []
for K in Ks:
    for s in Sigmas:
        # print("\t Testing s = {}".format(s))

        # kernel = RBFKernel(sigma=s,scale=1)
        # fastLogDet = FastIVM(K, kernel, 1.0)
        
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

n_cores = 15
print("Running {} on {} cores".format(len(runs), n_cores))
results = Parallel(n_jobs=n_cores)(delayed(eval)(options = options, X = X) for options, X in runs)
df = pd.DataFrame(results)
df.to_csv("results.csv",index=False)

    # print("Selecting {} represantatives via Greedy with python logdet".format(K))
    # res = evaluate_optimizer(Greedy(K, logdet), X)
    # print("\t fval:\t{} \n\t runtime:\t{} \n\n".format(res["fval"], res["runtime"]))

    # print()
    # print("=== STREAM PROCESSING ===")
    # print()

    # print("Selecting {} represantatives via Random".format(K))
    # res = evaluate_stream(Random(K, fastLogDet), X)
    # print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

    # print("Selecting {} represantatives via Sieve".format(K))
    # res = evaluate_stream(SieveStreaming(K, fastLogDet, 1.0, 0.01), X)
    # print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

    # print("Selecting {} represantatives via Sieve++".format(K))
    # res = evaluate_stream(SieveStreamingPP(K, fastLogDet, 1.0, 0.01), X)
    # print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))

    # print("Selecting {} represantatives via ThreeSieves".format(K))
    # res = evaluate_stream(ThreeSieves(K, fastLogDet, 1.0, 0.01, "sieve", 1000), X)
    # print("\t fval:\t{} \n \t runtime:\t{} \n \n".format(res["fval"], res["runtime"]))
