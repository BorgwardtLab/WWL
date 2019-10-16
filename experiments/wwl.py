# -----------------------------------------------------------------------------
# This file contains the functions to compute the node embeddings and to compute
# the wasserstein distance matrix
#
# October 2019, M. Togninalli, E. Ghisu, B. Rieck
# -----------------------------------------------------------------------------
import numpy as np

from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin

import argparse
import igraph as ig
import os
import ot

import copy
from collections import defaultdict
from typing import List

from utilities import load_continuous_graphs, create_labels_seq_cont, retrieve_graph_filenames

####################
# Embedding schemes
####################
def compute_wl_embeddings_continuous(data_directory, h):
    '''
    Continuous graph embeddings
    TODO: for package implement a class with same API as for WL
    '''
    node_features, adj_mat, n_nodes = load_continuous_graphs(data_directory)
    
    node_features_data = scale(np.concatenate(node_features, axis=0), axis = 0)
    splits_idx = np.cumsum(n_nodes).astype(int)
    node_features_split = np.vsplit(node_features_data,splits_idx)		
    node_features = node_features_split[:-1]

    # Generate the label sequences for h iterations
    labels_sequence = create_labels_seq_cont(node_features, adj_mat, h)

    return labels_sequence

def compute_wl_embeddings_discrete(data_directory, h):
    graph_filenames = retrieve_graph_filenames(data_directory)
    
    graphs = [ig.read(filename) for filename in graph_filenames]

    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform(graphs, h)

    # Each entry in the list represents the label sequence of a single
    # graph. The label sequence contains the vertices in its rows, and
    # the individual iterations in its columns.
    #
    # Hence, (i, j) will contain the label of vertex i at iteration j.
    label_sequences = [
        np.full((len(graph.vs), h + 1), np.nan) for graph in graphs
    ]   

    for iteration in sorted(label_dicts.keys()):
        for graph_index, graph in enumerate(graphs):
            labels_raw, labels_compressed = label_dicts[iteration][graph_index]

            # Store label sequence of the current iteration, i.e. *all*
            # of the compressed labels.
            label_sequences[graph_index][:, iteration] = labels_compressed

    return label_sequences

####################
# Weisfeiler-Lehman
####################
class WeisfeilerLehman(TransformerMixin):
    """
    Class that implements the Weisfeiler-Lehman transform
    Credits: Christian Bock and Bastian Rieck
    """
    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: List[ig.Graph]):
        num_unique_labels = 0
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
            
            if not 'label' in x.vs.attribute_names():
                x.vs['label'] = list(map(str, [l for l in x.vs.degree()]))           
            labels = x.vs['label']
            

            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels.append(self._preprocess_relabel_dict[label])
            x.vs['label'] = new_labels
            self._results[0][i] = (labels, new_labels)
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X: List[ig.Graph], num_iterations: int=3):
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = g.vs['label']

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)
                self._relabel_steps[i][it] = { idx: {old_label: new_labels[idx]} for idx, old_label in enumerate(current_labels) }
                g.vs['label'] = new_labels

                self._results[it][i] = (merged_labels, new_labels)
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        return self._results

    def _relabel_graph(self, X: ig.Graph, merged_labels: list):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str,merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[list]):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X: ig.Graph, sort: bool=True):
            neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
            neighbor_labels = []
            for n_indices in neighbor_indices:
                if sort:
                    neighbor_labels.append( sorted(X.vs[n_indices]['label']) )
                else:
                    neighbor_labels.append( X.vs[n_indices]['label'] )
            return neighbor_labels


def compute_wasserstein_distance(label_sequences, h, sinkhorn=False, 
                                    discrete=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n = len(label_sequences)
    emb_size = label_sequences[0].shape[1]
    n_feat = int(emb_size/(h+1))

    # Iterate over all possible h to generate the Wasserstein matrices
    hs = range(0, h + 1)
    
    wasserstein_distances = []
    for h in hs:
        M = np.zeros((n,n))
        # Iterate over pairs of graphs
        for graph_index_1, graph_1 in enumerate(label_sequences):
            # Only keep the embeddings for the first h iterations
            labels_1 = label_sequences[graph_index_1][:,:n_feat*(h+1)]
            for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
                labels_2 = label_sequences[graph_index_2 + graph_index_1][:,:n_feat*(h+1)]
                # Get cost matrix
                ground_distance = 'hamming' if discrete else 'euclidean'
                costs = ot.dist(labels_1, labels_2, metric=ground_distance)

                if sinkhorn:
                    mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                        np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                        numItermax=50)
                    M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
                else:
                    M[graph_index_1, graph_index_2 + graph_index_1] = \
                        ot.emd2([], [], costs)
                        
        M = (M + M.T)
        wasserstein_distances.append(M)
        print(f'Iteration {h}: done.')
    return wasserstein_distances
