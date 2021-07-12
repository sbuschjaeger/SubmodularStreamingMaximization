
![Submodular Streaming Maximization](docs/pyssm-logo-bw.png "Submodular Streaming Maximization")

[![Building docs](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/generate_docs.yml/badge.svg)](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/generate_docs.yml) 
[![Python tests](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/test_python.yml/badge.svg)](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/test_python.yml)
[![C++ tests](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/test_cpp.yml/badge.svg)](https://github.com/sbuschjaeger/SubmodularStreamingMaximization/actions/workflows/test_cpp.yml)

**18.06.2021**: Our paper has been accepted at the ECML/PKDD 2021. Stay tuned for an updated version. 🤩

This repository includes the code for our paper [Very Fast Streaming Submodular Function Maximization](https://arxiv.org/abs/2010.10059) which introduces a new  nonnegative submodular function maximization algorithm for streaming data. For our experiments, we also implemented already existing state-of-the-art streaming algorithms for which we could not find an implementation. The code focuses on easy extensibility and accessibility. It is mainly written in header-only C++ with a Python interface via pybind11. A detailed documentation can be found [here](https://sbuschjaeger.github.io/SubmodularStreamingMaximization/html/root.html). 

## How to use this code

First clone this repo recursively

    git clone --recurse-submodules git@github.com:sbuschjaeger/SubmodularStreamingMaximization.git 

### Using the Python Interface

If you use anaconda to manage your python packages you can use the following commands to install all dependencies including the python interface `PySSM` 
    
    source /opt/anaconda3/bin/activate 
    conda env create -f environment.yml  --force
    conda activate pyssm

Note the `--force` option which overrides all existing environments called `pyssm`. Alternatively, you can just install this package via

    pip install -e .

Once installed, you can simply import the desired submodular function and optimizer via `PySSM`. For a detailed explanation on specific parameters / functions provided please have a look at the documentation of the source code.
The following example uses the Greedy optimizer to select a data summary by maximizing the Informative Vector Machine (the full examples can be found in `tests/main.py`)

```python
from PySSM import RBFKernel
from PySSM import IVM
from PySSM import Greedy

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
kernel = RBFKernel(sigma=1,scale=1)
ivm = IVM(kernel = kernel, sigma = 1.0)
greedy = Greedy(K, ivm)

greedy.fit(X)

# Alternativley, you can use the streaming interface. 
#for x in X:
#    opt.next(x)

fval = opt.get_fval()
solution = np.array(opt.get_solution())

print("Found a solution with fval = {}".format(fval))
print(solution)
```

### Using the C++ interface

The C++ code is header-only so simply include the desired functions in your project and your are good to go. However, you require a C++17 compiler (e.g. gcc-7 or clang-5). If you have trouble compiling you can look at the `CMakeLists.txt` file which compiles the Python bindings as well as the following test file. The following example uses the Greedy optimizer to select a data summary by maximizing the Informative Vector Machine (the full examples can be found in `tests/main.cpp`)

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include "FastIVM.h"
#include "RBFKernel.h"
#include "Greedy.h"

std::vector<std::vector<double>> data = {
    {0, 0},
    {1, 1},
    {0.5, 1.0},
    {1.0, 0.5},
    {0, 0.5},
    {0.5, 1},
    {0.0, 1.0},
    {1.0, 0.0}
};    

unsigned int K = 3;
FastIVM fastIVM(K, RBFKernel(), 1.0);

Greedy greedy(K, fastIVM)
greedy.fit(data);
auto solution = greedy.get_solution();
double fval = greedy.get_fval();

std::cout << "Found a solution with fval = " << fval << std::endl;
for (auto x : solution) {
    for (auto xi : x) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;
}
```

## How to reproduce the experiments in our paper

This repository combines code for multiple datasets and two different repositories. It is structured as the following:

- `experiment_runner`: Contains code for the `experiment_runner` package. This is a small tool which we use to run multiple experiments in our computing environment. Basically, the `experiment_runner` accepts a list of experiments and runs these either on multiple machines or in a multi-threaded environment. All results are written to a `*.jsonl`. This code is sparsely documented. For a commented example please have a look at `experiments/kdd99/run.{cpp,py}`.
- `include/`: Contains the `C++` implementation of all optimizers. You can find more documentation under `docs/html/index.html` including some examples how to use specific optimizers.
- `tests/run.{py,cpp}`: Contains sample code for `C++` and `Python`
- `DATASET/run.py`: Contains code to run the experiments on the specific DATASET. 
- `DATASET/init.sh`: Some datasets must be downloaded manually. In this case you can use the provided `init.sh` to do so.
- `explore_results.ipynb`: A juypter notebook to plot the results of each experiment

All experiments have been performed via the Python interface. Assuming that you followed the instructions above, simply navigate to the specific folder and execute the run file
    
    cd experiments/kddcup99
    ./run.py

In the `experiments` folder you can find code which runs experiments on various dataset via the Python interface. You will probably need to download the data first which can be done using the `init.{sh,py}` scripts provided in each folder. Some notes on this:

- `creditfraud` is hosted on kaggle, which requires the kaggle-api to be installed and configured with an API key. It might be easier to manually download this data-set from kaggle
- `fact-highlevel` is hosted by the FACT project page. To process these files some additional packages are required which should be installed via the conda environment. If not, please make sure to have `pyfact` installed and working. Also note, that these files are rather large (> 2GB) so the download may take some time.
- `fact-lowlevel` requires even more additional tools and packages. Please contact sebastian.buschjaeger@tu-dortmund.de if you are interested in these files.

Once the data is downloaded, you can start the experiments by executing `run.py` in the respective folder. This file starts _all_ experiments for a single data-set including all hyperparameter configurations. This may take some time (usually a few hours per data-set) to finish. The experiments are currently configured to launch `15` threads via `joblib`, so make sure your hardware is strong enough or reduce the number of cores by setting `n_cores` at the end of each file. After the experiments finished, you can browse the results by using the `explore_results` Jupyter Notebook. Note that, depending on actual experiments you ran you might want to change the list of `datasets` used for plotting in the second cell of this notebook accordingly.

## Citing our Paper

    @misc{buschjäger2020fast,
          title={Very Fast Streaming Submodular Function Maximization}, 
          author={Sebastian Buschjäger and Philipp-Jan Honysz and Katharina Morik},
          year={2020},
          eprint={2010.10059},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
