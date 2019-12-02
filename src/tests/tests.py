import numpy as np
import os
from wwl import *
import igraph as ig

# Hard paths for the files
data_folder = '../data'
output_folder = '../experiments/output'

def test_categorical_embedding():
    # Check if embeddings for the 
    dataset = 'MUTAG'
    graphs = [ig.read(g) for g in retrieve_graph_filenames(os.path.join(data_folder,dataset))]
    # Embed
    wl = WeisfeilerLehman()
    node_representations = wl.fit_transform(graphs, num_iterations=2)
    # load ground truth
    gt = np.load(os.path.join(output_folder, dataset, 'MUTAG_wl_discrete_embeddings_h2.npy'))
    for g1,g2 in zip(node_representations, gt):
        assert np.allclose(g1,g2)

def test_continuous_embeddings():
    # Check if embeddings for the 
    dataset = 'ENZYMES'
    graphs = [ig.read(g) for g in retrieve_graph_filenames(os.path.join(data_folder,dataset))]
    node_features = np.load(os.path.join(data_folder,dataset,'node_features.npy'))
    # Embed
    wl = ContinuousWeisfeilerLehman()
    node_representations = wl.fit_transform(graphs, node_features=node_features, num_iterations=2)
    # load ground truth
    gt = np.load(os.path.join(output_folder, dataset, f'ENZYMES_wl_continuous_embeddings_h2.npy'))
    for i in range(len(node_representations)):
        assert np.allclose(node_representations[i],gt[i])

def test_categorical_wasserstein_dist():
    # Check if embeddings for the 
    dataset = 'MUTAG'
    graphs = [ig.read(g) for g in retrieve_graph_filenames(os.path.join(data_folder,dataset))]
    # Embed and compute distance
    dist = pairwise_wasserstein_distance(graphs, num_iterations=2)
    # load ground truth
    gt = np.load(os.path.join(output_folder, dataset, 'wasserstein_distance_matrix_it2.npy'))
    assert np.allclose(dist,gt)

def test_continuous_wasserstein_dist():
    # Check if embeddings for the 
    dataset = 'ENZYMES'
    graphs = [ig.read(g) for g in retrieve_graph_filenames(os.path.join(data_folder,dataset))]
    # Embed and compute distance
    node_features = np.load(os.path.join(data_folder,dataset,'node_features.npy'))
    dist = pairwise_wasserstein_distance(graphs, node_features=node_features, num_iterations=2)
    # load ground truth
    gt = np.load(os.path.join(output_folder, dataset, 'wasserstein_distance_matrix_it2.npy'))
    assert np.allclose(dist,gt)