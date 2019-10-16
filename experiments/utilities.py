# -----------------------------------------------------------------------------
# This file contains several utility functions for reproducing results 
# of the WWL paper
#
# October 2019, M. Togninalli
# -----------------------------------------------------------------------------
import numpy as np
import os
import igraph as ig

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score

#################
# File loaders 
#################
def read_labels(filename):
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels

def read_gml(filename):
	node_features = []
	g = ig.read(filename)
		
	if not 'label' in g.vs.attribute_names():
		g.vs['label'] = list(map(str, [l for l in g.vs.degree()]))    
	
	node_features = g.vs['label']

	adj_mat = np.asarray(g.get_adjacency().data)
	
	return node_features, adj_mat

def retrieve_graph_filenames(data_directory):
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gml')]
    graphs.sort()
    return [os.path.join(data_directory, g) for g in graphs]

def load_continuous_graphs(data_directory):
    graph_filenames = retrieve_graph_filenames(data_directory)

    # initialize
    node_features = []
    adj_mat = []
    n_nodes = []

    # Iterate across graphs and load initial node features
    for graph_fname in graph_filenames:
        node_features_cur, adj_mat_cur = read_gml(graph_fname)
        # Load features
        node_features.append(np.asarray(node_features_cur).astype(float).reshape(-1,1))
        adj_mat.append(adj_mat_cur.astype(int))
        n_nodes.append(adj_mat_cur.shape[0])

    # Check if there is a node_features.npy file 
    # containing continuous attributes
    # PS: these were obtained by processing the TU Dortmund website
    # If none is present, keep degree or label as features.
    attribtues_filenames = os.path.join(data_directory, 'node_features.npy')
    if os.path.isfile(attribtues_filenames):
        node_features = np.load(attribtues_filenames)

    n_nodes = np.asarray(n_nodes)
    node_features = np.asarray(node_features)

    return node_features, adj_mat, n_nodes


def load_matrices(directory):
    '''
    Loads all the wasserstein matrices in the directory.
    '''
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    wass_matrices = []
    hs = []
    for f in sorted(files):
        hs.append(int(f.split('.npy')[0].split('it')[-1])) # Hoping not to have h > 9 !
        wass_matrices.append(np.load(os.path.join(directory,f)))
    return wass_matrices, hs

##################
# Graph processing
##################

def create_adj_avg(adj_cur):
	'''
	create adjacency
	'''
	deg = np.sum(adj_cur, axis = 1)
	deg = np.asarray(deg).reshape(-1)

	deg[deg!=1] -= 1

	deg = 1/deg
	deg_mat = np.diag(deg)
	adj_cur = adj_cur.dot(deg_mat.T).T
	
	return adj_cur


def create_labels_seq_cont(node_features, adj_mat, h):
	'''
	create label sequence for continuously attributed graphs
	'''
	n_graphs = len(node_features)
	labels_sequence = []
	for i in range(n_graphs):
		graph_feat = []

		for it in range(h+1):
			if it == 0:
				graph_feat.append(node_features[i])
			else:
				adj_cur = adj_mat[i]+np.identity(adj_mat[i].shape[0])
				adj_cur = create_adj_avg(adj_cur)

				np.fill_diagonal(adj_cur, 0)
				graph_feat_cur = 0.5*(np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
				graph_feat.append(graph_feat_cur)

		labels_sequence.append(np.concatenate(graph_feat, axis = 1))
		if i % 100 == 0:
			print(f'Processed {i} graphs out of {n_graphs}')
	
	return labels_sequence


#######################
# Hyperparameter search
#######################

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]
