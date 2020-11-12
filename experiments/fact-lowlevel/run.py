#!/usr/bin/env python3

import os
import random
from scipy.io import arff
import numpy as np
import pandas as pd

import numpy as np
from numpy.linalg import slogdet
import time
from experiment_runner.experiment_runner_v2 import run_experiments

# from PySSM import Matrix, Vector
from PySSM import RBFKernel
from PySSM import IVM, FastIVM

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 
from PySSM import Salsa 
from PySSM import IndependentSetImprovement

import os
import numpy as np
import scipy.io
import scipy.io
from sklearn import preprocessing


def pre(cfg):
    name = cfg["method"]
    sigma = cfg["sigma"]
    scale = cfg["scale"]
    K = cfg["K"]

    kernel = RBFKernel(sigma=sigma,scale=scale)
    fastLogDet = FastIVM(K, kernel, 1.0)

    if name == "Greedy":
        opt = Greedy(K, fastLogDet)
    if name == "IndependentSetImprovement":
        opt = IndependentSetImprovement(K, fastLogDet)
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
    
    opt.fit(cfg["X"],cfg["K"])
    return opt

def post(cfg, opt):
    return {
        "fval":opt.get_fval(),
        "num_candidate_solutions":opt.get_num_candidate_solutions(),
        "num_elements_stored":opt.get_num_elements_stored(),
    }

print("Loading data")

path = "/home/share02/datensaetze/fact_icdm_2020/crab/factExamples_no_cut_20131101_embedded_256.npy"
data = np.load(path)
idx = np.random.choice(range(0, data.shape[0]), 200000)
X = data[idx,:]

# MinMax normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

Ks = range(5,105,5)
# Ks = [5]
eps = [1e-1, 5e-2, 1e-2, 1e-3, 5e-3]
Ts = [500, 1000, 2500, 5000]
#Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])
Sigmas = [np.sqrt(X.shape[1])]

basecfg = {
    "out_path":"results",
    "backend":"multiprocessing",
    "num_cpus":10,
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
                "method": "IndependentSetImprovement",
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

random.shuffle(runs)
run_experiments(basecfg, runs)
