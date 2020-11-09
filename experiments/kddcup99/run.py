#!/usr/bin/env python3

import os
from scipy.io import arff
import numpy as np
import pandas as pd

import numpy as np
from numpy.linalg import slogdet
import time

from experiment_runner.experiment_runner_v2 import run_experiments

#from joblib import Parallel, delayed

# from PySSM import Matrix, Vector
from PySSM import RBFKernel
from PySSM import IVM, FastIVM

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 
from PySSM import Salsa 

def pre(cfg):
    name = cfg["method"]
    sigma = cfg["sigma"]
    scale = cfg["scale"]
    K = cfg["K"]

    kernel = RBFKernel(sigma=sigma,scale=scale)
    fastLogDet = FastIVM(K, kernel, 1.0)

    if name == "Greedy":
        opt = Greedy(K, fastLogDet)
    elif name == "Random":
        opt = Random(K, fastLogDet, cfg["run_id"])
    elif name == "SieveStreaming":
        e = cfg["epsilon"]
        opt = SieveStreaming(K, fastLogDet, 1.0, e)
    elif name == "SieveStreaming++":
        e = cfg["epsilon"]
        opt = SieveStreamingPP(K, fastLogDet, 1.0, e)
    elif name == "Salsa":
        e = cfg["epsilon"]
        opt = Salsa(K, fastLogDet, 1.0, e)
    elif name == "ThreeSieves":
        e = cfg["epsilon"]
        T = cfg["T"]
        opt = ThreeSieves(K, fastLogDet, 1.0, e, "sieve", T)
    return opt

def fit(cfg, opt):
    X = cfg["X"]
    
    opt.fit(cfg["X"], cfg["K"])
    return opt

def post(cfg, opt):
    return {"fval":opt.get_fval()}

print("Loading data")
data, meta = arff.loadarff(os.path.join(os.path.dirname(__file__), "data", "KDDCup99", "KDDCup99_withoutdupl_norm_1ofn.arff"))

data_pd = pd.DataFrame(data)
data_pd.columns = meta

y = np.array([-1 if x == "yes" else 1 for x in data_pd["outlier"]])  # 1 = inlier, -1 = outlier

data_pd = data_pd.drop("outlier", axis=1)
data_pd = data_pd.drop("id", axis=1)

X = data_pd.values

Ks = range(5,10,5)
# Ks = [5]
eps = [1e-3,5e-3,1e-2,5e-2,1e-1]
Ts = [500, 1000, 2500, 5000]
#Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])
Sigmas = [np.sqrt(X.shape[1])]

basecfg = {
    "out_path":"results",
    "backend":"multiprocessing",
    "num_cpus":5,
    "pre": pre,
    "post": post,
    "fit": fit,
}

results = []

runs = []
for K in Ks:
    for s in Sigmas:
        runs.append(
            ({   
                "method": "Greedy",
                "K":K,
                "sigma":s,
                "scale":1,
                "X":X
            })
        )

        runs.append(
            ({   
                "method": "Random",
                "K":K,
                "sigma":s,
                "scale":1,
                "repetitions":5,
                "X":X
            })
        )

        for e in eps:
            runs.append(
                ( {   
                    "method": "SieveStreaming",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "epsilon":e,
                    "X":X
                })
            )

            runs.append(
                ( {   
                    "method": "SieveStreaming++",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "epsilon":e,
                    "X":X
                })
            )

            runs.append(
                ( {   
                    "method": "Salsa",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "epsilon":e,
                    "X":X
                })
            )

            for T in Ts:    
                runs.append(
                    ( {   
                        "method": "ThreeSieves",
                        "K":K,
                        "sigma":s,
                        "scale":1,
                        "epsilon":e,
                        "T":T,
                        "X":X
                    })
                )

run_experiments(basecfg, runs)
# n_cores = 15
# print("Running {} on {} cores".format(len(runs), n_cores))
# results = Parallel(n_jobs=n_cores)(delayed(eval)(options = options, X = X) for options, X in runs)
# df = pd.DataFrame(results)
# df.to_csv("results.csv",index=False)
