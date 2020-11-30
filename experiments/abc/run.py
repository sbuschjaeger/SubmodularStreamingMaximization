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
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

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
    X = np.load("/home/share/fuerBuschjaeger/threesieves/abc/abc.npy")

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    if cfg["method"] == "Greedy":
        opt.fit(X,1)
    else:
        for x in X:
            opt.next(x)
    
    return opt

def post(cfg, opt):
    solution_dict = {
        "name" : cfg.get("method", None),
        "sigma" : cfg.get("sigma", None),
        "scale" : cfg.get("scale", None),
        "K" : cfg.get("K", None),
        "epsilon":cfg.get("epsilon", None),
        "T":cfg.get("T", None),
        "run_id":cfg.get("run_id", None),
        "out_path":cfg.get("out_path", None),
        "solution":opt.get_solution(),
        "fval":opt.get_fval()
    }

    np.save(cfg["out_path"],solution_dict,allow_pickle=True)

    return {
        "fval":opt.get_fval(),
        "num_candidate_solutions":opt.get_num_candidate_solutions(),
        "num_elements_stored":opt.get_num_elements_stored(),
    }

X = np.load("/home/share/fuerBuschjaeger/threesieves/abc/abc.npy")

Ks = range(5,105,5)
# Ks = [5]
eps = [1e-1, 1e-2] #, 1e-3
Ts = [250, 500, 1000, 1500, 2000, 2500, 5000]
#Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])
Sigmas = [0.25*np.sqrt(X.shape[1]), 0.5*np.sqrt(X.shape[1]), np.sqrt(X.shape[1]), 2*np.sqrt(X.shape[1])]

basecfg = {
    "out_path":"results",
    "backend":"ray",
    "address":"129.217.30.245:6379",
    "redis_password":"5241590000000000",
    "num_cpus":1,
    "max_memory":6*1024*1024*1024, # 6 GB
    "pre": pre,
    "post": post,
    "fit": fit
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
            })
        )

        runs.append(
            ({   
                "method": "IndependentSetImprovement",
                "K":K,
                "sigma":s,
                "scale":1,
            })
        )

        runs.append(
            ({   
                "method": "Random",
                "K":K,
                "sigma":s,
                "scale":1,
                "repetitions":5,
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
                })
            )

            runs.append(
                ( {   
                    "method": "SieveStreaming++",
                    "K":K,
                    "sigma":s,
                    "scale":1,
                    "epsilon":e,
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
                    })
                )

run_experiments(basecfg, runs)
