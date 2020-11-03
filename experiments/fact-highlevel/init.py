#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import sys
import time
import urllib
import os.path

from struct import pack

from sklearn import preprocessing

import numpy as np
import pandas as pd
import fact.io
import progressbar

N = 200000
factLevel2URL = "https://factdata.app.tu-dortmund.de/dl2/FACT-Tools/v1.1.1/"
gammaName = "gamma_simulations_facttools_dl2.hdf5"
protonName = "proton_simulations_facttools_dl2.hdf5"
outFolder = "./"


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

if not os.path.isfile(outFolder + "/" + gammaName):
    print("Fact gamma data not found, downloading it from", factLevel2URL + "\n")
    urllib.request.urlretrieve(factLevel2URL + gammaName, outFolder + "/" + gammaName, reporthook)

if not os.path.isfile(outFolder + "/" + protonName):
    print("Fact proton data not found, downloading it from", factLevel2URL + "\n")
    urllib.request.urlretrieve(factLevel2URL + protonName, outFolder + "/" + protonName, reporthook)

print("Loading ", outFolder + "/" + gammaName, " with FACT tools")
#gammaDF = fact.io.read_data(outFolder + "/" + gammaName, key='events', first=_first+shift, last=_last+shift)
gammaDF = fact.io.read_data(outFolder + "/" + gammaName, key='events')
gammaDF["label"] = False

#print("{} item in gammaDF".format(len(gammaDF)))

print("Loading ", outFolder + "/" + protonName, " with FACT tools")
protonDF = fact.io.read_data(outFolder + "/" + protonName, key='events')
protonDF["label"] = True

#print("{} items in protonDF".format(len(protonDF)))

#asdf
header = [
    "concentration_cog",
    "concentration_core",
    "concentration_one_pixel",
    "concentration_two_pixel",
    "leakage1",
    "leakage2",
    "size",
    "width",
    "length",
    "skewness_long",
    "skewness_trans",
    "kurtosis_long",
    "kurtosis_trans",
    "num_islands",
    "num_pixel_in_shower",
    "photoncharge_shower_mean",
    "photoncharge_shower_variance",
    "area",
    "log_size",
    "log_length",
    "size_area",
    "area_size_cut_var"
]

#print(protonDF.columns.tolist())
#gammaDF = gammaDF[gammaDF.columns.tolist()]
qualityCuts = "num_pixel_in_shower >= 10 & num_islands < 8 & length < 70 & width < 35 & leakage1 < 0.6 & leakage2 < 0.85"

print("Applying sanity filtering")
#gammaDF = gammaDF.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
#protonDF = protonDF.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

print("Applying quality cuts")
gammaDF = gammaDF.query(qualityCuts)
protonDF = protonDF.query(qualityCuts)

print("Num gammas after filtering+cuts: ", len(gammaDF))
print("Num protons after filtering+cuts: ", len(protonDF))	
print("Generate some additional features taken from Jens' YML script")

# There is a "chained assignment" warning for the following computations
# However, I dont see a reason why this should be a problem. Thus we disable the warning
pd.options.mode.chained_assignment = None

gammaDF["area"] = gammaDF["width"]*gammaDF["length"]*np.pi
gammaDF["log_size"] = np.log(gammaDF["size"])
gammaDF["log_length"] = np.log(gammaDF["length"])	
gammaDF["size_area"] = gammaDF["size"]/(gammaDF["width"]*gammaDF["length"]*np.pi)
gammaDF["area_size_cut_var"] = (gammaDF["width"]*gammaDF["length"]*np.pi) /(np.log(gammaDF["size"])**2)

protonDF["area"] = protonDF["width"]*protonDF["length"]*np.pi
protonDF["log_size"] = np.log(protonDF["size"])
protonDF["log_length"] = np.log(protonDF["length"])	
protonDF["size_area"] = protonDF["size"]/(protonDF["width"]*protonDF["length"]*np.pi)
protonDF["area_size_cut_var"] = (protonDF["width"]*protonDF["length"]*np.pi) /(np.log(protonDF["size"])**2)

print("Selecting rows")
gammaDF = gammaDF[header + ["label"]]
protonDF = protonDF[header + ["label"]]

print("Shuffle data")
gammaDF = gammaDF.sample(frac=1).reset_index(drop=True)
protonDF = protonDF.sample(frac=1).reset_index(drop=True)

print("Sampling data")
#dfTrain = pd.concat([gammaDF.sample(n=int(NTrain/2)), protonDF.sample(n=int(NTrain/2))])
df = pd.concat([gammaDF.loc[0:int(N/2)-1,:], protonDF.loc[0:int(N/2)-1,:]])
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("data.csv", sep=",", index=False, header=True)
