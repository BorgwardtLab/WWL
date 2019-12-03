# Wasserstein Weisfeiler-Lehman Graph Kernels
This repository contains the accompanying code for the NeurIPS 2019 paper
_Wasserstein Weisfeiler-Lehman Graph Kernels_ available 
[here](http://papers.nips.cc/paper/8872-wasserstein-weisfeiler-lehman-graph-kernels).
The repository contains both the package that implements the graph kernels (in `src`)
and scripts to reproduce some of the results of the paper (in `experiments`).

## Dependencies

WWL relies on the following dependencies:

- `numpy`
- `scikit-learn`
- `POT`
- `cython`

## Installation

The easiest way is to install WWL from the Python Package Index (PyPI) via

```sh
$ pip install Cython numpy 
$ pip install wwl
```

## Usage

The WWL package contains functions to generate a `n x n` kernel matrix between 
a set of `n` graphs.

The API also allows the user to directly call the different steps described in the paper, namely:
- generate the embeddings for the nodes of both discretely labelled and continuously attributed graphs,
- compute the pairwise distance between a set of graphs

Please refer to [the src README](https://github.com/BorgwardtLab/WWL/blob/master/src) for detailed documentation.


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
