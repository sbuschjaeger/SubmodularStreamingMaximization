#!/usr/bin/env python3

import os
import arff
import numpy as np
import pandas

import numpy as np
from numpy.linalg import slogdet

from PySSM import RBFKernel
from PySSM import IVM, FastIVM
from PySSM import SubmodularFunction

from PySSM import Greedy
from PySSM import Random
from PySSM import SieveStreaming
from PySSM import SieveStreamingPP
from PySSM import ThreeSieves 


data = arff.load(open(os.path.join(os.path.dirname(__file__), "data", "KDDCup99", "KDDCup99_withoutdupl_norm_1ofn.arff"), "r"))
data_pd = pandas.DataFrame(data["data"])
data_pd.columns = [x[0] for x in data["attributes"]]

# Extract label vector
y = np.array([-1 if x == "yes" else 1 for x in data_pd["outlier"]])  # 1 = inlier, -1 = outlier

# Delete irrelevant features.
data_pd = data_pd.drop("outlier", axis=1)
data_pd = data_pd.drop("id", axis=1)

# Only values from now on
X = data_pd.values

K = 50
fastLogDet = FastIVM(K)
opt = Greedy(K, fastLogDet)
opt.fit(X)

fval = opt.get_fval()
solution = np.array(opt.get_solution())

print("Found a solution with fval = {}".format(fval))
print(solution)
