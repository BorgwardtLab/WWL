'''Wasserstein Weisfeiler-Lehman Graph Kernels'''

__version__ = '0.1.2'
from .wwl import wwl, pairwise_wasserstein_distance
from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman
from .utilities import wwl_custom_grid_search_cv, retrieve_graph_filenames