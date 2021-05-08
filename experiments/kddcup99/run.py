#!/usr/bin/env python3

import os
import random
from scipy.io import arff
import numpy as np
import pandas as pd
import argparse

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
from PySSM import IndependentSetImprovement

'''
In this script we compute data summaries on the kddcup99 dataset. This script configures _all_ experiments for this specific dataset.

We configure an experiment by three common functions which are called via the experiment_runner wrapper and some additional base information. 
Each experiment is configured through a dictionary. Apart from a few reserved keywords (e.g. pre/post/fit) every other field is simply passed to 
to each function and can be used as desired. The three functions are
    - pre: Everything which needs to be done before running the experiments
    - fit: Running the actual experiments
    - post: Everything which needs to be done after the experiment
'''

# This function is called by the experiment_runner to prepare an experiment.
# It expects each experiment configuration to contain a "method" field which is the c'tor of optimizer to-be-tested. 
# Moreover it extracts common parameters such as K, sigma, scaling for the kernel etc. 
def pre(cfg):
    name = cfg["method"]
    sigma = cfg["sigma"]
    scale = cfg["scale"]
    K = cfg["K"]

    # Create function to be maximized
    kernel = RBFKernel(sigma=sigma,scale=scale)
    fastLogDet = FastIVM(K, kernel, 1.0)

    # Create optimizer
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

# Fit the optimizer and extract the summary
def fit(cfg, opt):
    X = cfg["X"]
    
    opt.fit(cfg["X"], cfg["K"])
    return opt

# This function computes statistics after the summary has been computed.
def post(cfg, opt):
    return {
        "fval":opt.get_fval(),
        "num_candidate_solutions":opt.get_num_candidate_solutions(),
        "num_elements_stored":opt.get_num_elements_stored(),
    }

# Load the data and remove lables
print("Loading data")
data, meta = arff.loadarff(os.path.join(os.path.dirname(__file__), "data", "KDDCup99", "KDDCup99_withoutdupl_norm_1ofn.arff"))

data_pd = pd.DataFrame(data)
data_pd.columns = meta
data_pd = data_pd.drop("outlier", axis=1)
data_pd = data_pd.drop("id", axis=1)
X = data_pd.values

# Define parameters
Ks = range(5,105,5)
eps = [1e-1, 5e-2, 1e-2, 1e-3, 5e-3]
Ts = [500, 1000, 2500, 5000]
Sigmas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])*np.sqrt(X.shape[1])

# Gather some input configuration
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--single", help="Run experiments in a single thread",action="store_true", default=True)
parser.add_argument("-n", "--n_jobs", help="Number of jobs if --single is False",action="store_true", default=10)

args = parser.parse_args()


'''
Check if we single or multi-threaded
basecfg = {
    "out_path": Path where results should be stored
    "pre": The pre-function
    "post": The post-function
    "fit": The fit-function
    "backend": The running mode-, e.g. "local"/"ray"/"multiprocessing"
    "param_1": Additional parameter 1 required by the backend, e.g. the "ray_head"
    "param_2": Additional parameter 2 required by the backend, e.g. the "redis_password" etc
}
'''
if args.single:
    basecfg = {
        "out_path":"results",
        "backend":"local",
        "num_cpus":1,
        "pre": pre,
        "post": post,
        "fit": fit,
    }
else:
    basecfg = {
        "out_path":"results",
        "backend":"local",
        "num_cpus":args.n_jobs,
        "pre": pre,
        "post": post,
        "fit": fit
    }


# Configure all the runs
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

        runs.append(
            ({   
                "method": "IndependentSetImprovement",
                "K":K,
                "sigma":s,
                "scale":1,
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

# Execute the runs
random.shuffle(runs)
run_experiments(basecfg, runs)
