SubmodularStreamMaximization
============================

This repository contains code for Submodular Streaming Maximization. The code focuses on easy extensibility and accessibility. It is mainly written in header-only C++ with a Python interface via pybind11. Currently implemented algorithms are:


* :class:`Greedy`
* :class:`SieveStreaming`
* :class:`SieveStreamingPP`
* :class:`ThreeSieves`
* :class:`Random`
* :class:`Salsa`
* :class:`IndependentSetImprovement`

You can implement your own submodular functions in :doc:`Python <python>` or :doc:`C++ <cpp>` and pass them to the optimizers. In addition, the following functions are currently implemented:

* Informative Vector Machine (:class:`IVM`) (sometimes called LogDet) with a custom kernel. 
.. math::
   f(S) = \frac{1}{2}\log\det\left(\Sigma + \sigma \cdot \mathcal I \right)
where :math:`\Sigma = [k(x_i,x_j)]_{i,j}` is the kernel matrix, :math:`k(\cdot, \cdot)` is the kernel function, :math:`\sigma \in \mathbb R_{\ge 0}` is a scaling parameter and :math:`\mathcal I` is the :math:`K \times K` identity matrix. Look at this function if you want to implement your own submodular function as a simple example.

* Fast Informative Vector Machine (:class:`FastIVM`) with a custom kernel. This is the same as above, but much quicker. The implementation keeps track of the Cholesky Decomposition of the kernel matrix and updates it without re-computing the entire Kernelmatrix or its inverse. Use this function if speed is important. 

* :class:`RBFKernel` The RBF kernel for both IVM variants
.. math::
   k(x_j, x_j) = s \cdot \exp\left(- \frac{\|x_i - x_j \|_2^2}{\sigma}\right)
where :math:`s, \sigma \in \mathbb R_{\ge 0}` are scaling parameter. Look at this code if you want to implement your own kernel

How to install
--------------------------

First clone the repository:

.. code-block::

   git clone --recurse-submodules git@github.com:sbuschjaeger/SubmodularStreamingMaximization.git
   cd SubmodularStreamingMaximization

Note that we use the ``--recurse-submodules`` flag here to download additional dependencies. This projects depends on two other projects, namely `pybind <https://github.com/pybind/pybind11>`_ and the `experiment_runner <https://github.com/sbuschjaeger/experiment_runner>`_ . PyBind is used to provide the python interface, whereas the `experiment_runner <https://github.com/sbuschjaeger/experiment_runner>`_ is a small tool to run multiple experiments across different machines. If you are not interested in re-running the experiments for our paper (see below), then you can ignore this dependency. 

If you want to use the C++ interface you simply need to include the desired functions in your project and you are good to go. We use C++-17 (e.g. `std::optional`), so you need a somewhat recent compiler (e.g. gcc-7, clang-5). If you want to use the python interface then you need to install the package via

.. code-block::

   pip install -e .


If you use anaconda to manage your python packages you can use the following commands to install all dependencies including the python interface ``PySSM`` as well as the necessary compilers

.. code-block::

   conda env create -f environment.yml  --force
   conda activate pyssm

Note the ``--force`` option which overrides all existing environments called ``pyssm``. 

How to reproduce the experiments in our paper
---------------------------------------------

If you are interested in re-running the experiments for our `paper <https://arxiv.org/abs/2010.10059>`_ then this is easily possible. First, clone the repository as discussed above. Then you should end up with the following files:

- ``experiment_runner``: Contains code for the `experiment_runner <https://github.com/sbuschjaeger/experiment_runner>`_ package. This is a small tool which we use to run multiple experiments in our computing environment. Basically, this tool accepts a list of experiments and runs these either on multiple machines or in a multi-threaded environment. All results are written to a ``*.jsonl``. This code is sparsely documented. For a commented example please have a look at ``experiments/kdd99/run.{cpp,py}``.
- ``DATASET/run.py``: Contains code to run the experiments on the specific DATASET. 
- ``DATASET/init.{sh,py}``: Contains code to download the specific DATASET
- ``explore_results.ipynb``: A juypter notebook to plot the results of each experiment

All experiments have been performed via the Python interface. Before starting the experiments you will need to download the data first which can be done using the ``init.{sh,py}`` scripts provided in each folder. Some notes on this:

- ``creditfraud`` is hosted on kaggle, which requires the kaggle-api to be installed and configured with an API key. It might be easier to manually download this data-set from kaggle
- ``fact-highlevel`` is hosted by the FACT project page. To process these files some additional packages are required which should be installed via the conda environment. If not, please make sure to have `pyfact <https://github.com/fact-project/pyfact>`_ installed and working. Also note, that these files are rather large (> 2GB) so the download may take some time.
- ``fact-lowlevel`` requires even more additional tools and packages. Please contact `me <sebastian.buschjaeger@tu-dortmund.de>`_ if you are interested in these files.

Once the data is downloaded, you can start the experiments by executing ``run.py`` in the respective folder. This file starts **all** experiments for a single data-set including all hyperparameter configurations. This may take some time (usually a few hours per data-set) to finish. The experiments are currently configured to launch ``15`` threads via ``joblib``, so make sure your hardware is strong enough or reduce the number of cores by setting ``n_cores`` at the end of each file. A complete workflow looks like this

.. code-block::

    cd experiments/kddcup99
    ./init.sh
    ./run.py

After the experiments finished, you can browse the results by using the ``explore_results`` Jupyter Notebook. Note that, depending on actual experiments you ran you might want to change the list of ``datasets``  used for plotting in the second cell of this notebook accordingly.

Cite our paper
--------------

.. code-block::

    @misc{buschjäger2020fast,
          title={Very Fast Streaming Submodular Function Maximization}, 
          author={Sebastian Buschjäger and Philipp-Jan Honysz and Katharina Morik},
          year={2020},
          eprint={2010.10059},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }