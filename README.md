# Wasserstein Weisfeiler-Lehman Graph Kernels
This repository contains the accompanying code for the NeurIPS 2019 paper
_Wasserstein Weisfeiler-Lehman Graph Kernels_ available 
[here](https://arxiv.org/abs/1906.01277).
The repository contains both the package that implements the graph kernels (in `src`)
and scripts to reproduce some of the results of the paper (in `experiments`).

## Dependencies

WWL relies on the following dependencies:

- `numpy`
- `scikit-learn`
- `POT`
- `cython`

## Installation

**CURRENTLY UNDER DEVELOPMENT**

The easiest way is to install WWL from the Python Package Index (PyPI) via

```
$ pip install cython numpy wwl
```

## Usage

**CURRENTLY UNDER DEVELOPMENT**

The WWL package contains function to generate a `n x n` kernel matrix between 
a set of `n` graphs.

The API also allows the user to directly call the different steps described in the paper, namely:
- generate the embeddings for the nodes of both discretely labelled and continuously attributed graphs,
- 

The package provides functions to transform a set of `n` training time series and `o` test time series into an `n x n` distance matrix for training and an `o x n` distance matrix for testing.
Additionally, we provide a way to run a grid search for a krein space SVM. `krein_svm_grid_search` runs a `5`-fold
cross-validation on the training set to determine the best hyperparameters. Then, its classification accuracy is
computed on the test set.

## Experiments

You can find some experiments in the [experiments folder](https://github.com/BorgwardtLab/WWL/blob/master/experiments). These will allow you to reproduce results from the paper on 2 datasets.


## Contributors

WWL is developed and maintained by members of the [Machine Learning and
Computational Biology Lab](https://www.bsse.ethz.ch/mlcb):

- Matteo Togninalli ([GitHub](https://github.com/mtog))
- Elisabetta Ghisu ([Github](https://github.com/eghisu))
- Bastian Rieck ([GitHub](https://github.com/Pseudomanifold))

## Citation
Please use the following BibTeX citation when using our method or comparing against it:
```
@InCollection{Togninalli19,
  author        = {Togninalli, Matteo and Ghisu, Elisabetta and Llinares-López, Felipe and Rieck, Bastian and Borgwardt, Karsten},
  title         = {Wasserstein Weisfeiler--Lehman Graph Kernels},
  booktitle     = {Advances in Neural Information Processing Systems 32},
  year          = {2019},
  pubstate      = {forthcoming},
  eprint        = {1906.01277},
  archiveprefix = {arXiv},
  author+an     = {4=highlight},
  primaryclass  = {cs.LG},
}
```
