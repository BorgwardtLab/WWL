# -----------------------------------------------------------------------------
# This script runs the experiments reported in the WWL paper
#
# October 2019, M. Togninalli, E. Ghisu, B. Rieck
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import argparse
import os

from utilities import read_labels, custom_grid_search_cv
from wwl import *
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name (MUTAG or Enzymes)',
                            choices=['MUTAG', 'ENZYMES'])
    parser.add_argument('--crossvalidation', default=False, action='store_true', help='Enable a 10-fold crossvalidation')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('--h', type = int, required=False, default=2, help = "(Max) number of WL iterations")

    args = parser.parse_args()
    dataset = args.dataset
    h = args.h
    sinkhorn = args.sinkhorn
    print(f'Generating results for {dataset}...')
    #---------------------------------
    # Setup
    #---------------------------------
    # Start by making directories for intermediate and final files
    data_path = os.path.join('../data', dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)
    
    for path in [output_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    embedding_type = 'continuous' if dataset == 'ENZYMES' else 'discrete'
    print(f'Generating {embedding_type} embeddings for {dataset}.')
    if dataset == 'ENZYMES':
        label_sequences = compute_wl_embeddings_continuous(data_path, h)
    else:
        label_sequences = compute_wl_embeddings_discrete(data_path, h)

    # Save embeddings to output folder
    out_name = f'{dataset}_wl_{embedding_type}_embeddings_h{h}.npy'
    np.save(os.path.join(output_path, out_name), label_sequences)
    print(f'Embeddings for {dataset} computed, saved to {os.path.join(output_path, out_name)}.')
    print()

    #---------------------------------
    # Wasserstein & Kernel computations
    #---------------------------------
    # Run Wasserstein distance computation
    print('Computing the Wasserstein distances...')
    wasserstein_distances = compute_wasserstein_distance(label_sequences, h, sinkhorn=sinkhorn,
                                                            discrete=dataset=='MUTAG')

    # Save Wasserstein distance matrices
    for i, D_w in enumerate(wasserstein_distances):
        filext = 'wasserstein_distance_matrix'
        if sinkhorn:
            filext += '_sinkhorn'
        filext += f'_it{i}.npy'
        np.save(os.path.join(output_path,filext), D_w)
    print('Wasserstein distances computation done. Saved to file.')
    print()

    # Transform to Kernel
    # Here the flags come into play
    if args.gridsearch:
        # Gammas in eps(-gamma*M):
        gammas = np.logspace(-4,1,num=6)  
        # iterate over the iterations too
        hs = range(h)
        param_grid = [
            {'C': np.logspace(-3,3,num=7)}
        ]
    else:
        gammas = [0.001]
        hs = [h]

    kernel_matrices = []
    kernel_params = []
    for i, current_h in enumerate(hs):
        # Generate the full list of kernel matrices from which to select
        M = wasserstein_distances[current_h]
        for g in gammas:
            K = np.exp(-g*M)
            kernel_matrices.append(K)
            kernel_params.append((current_h, g))

    # Check for no hyperparam:
    if not args.gridsearch:
        assert len(kernel_matrices) == 1
    print('Kernel matrices computed.')
    print()

    #---------------------------------
    # Classification
    #---------------------------------
    # Run hyperparameter search if needed
    print(f'Running SVMs, crossvalidation: {args.crossvalidation}, gridsearch: {args.gridsearch}.')
    # Load labels
    label_file = os.path.join(data_path, 'Labels.txt')
    y = np.array(read_labels(label_file))

    # Contains accuracy scores for each cross validation step; the
    # means of this list will be used later on.
    accuracy_scores = []
    np.random.seed(42)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    # Hyperparam logging
    best_C = []
    best_h = []
    best_gamma = []

    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        # Gridsearch
        if args.gridsearch:
            gs, best_params = custom_grid_search_cv(SVC(kernel='precomputed'), 
                    param_grid, K_train, y_train, cv=5)
            # Store best params
            C_ = best_params['params']['C']
            h_, gamma_ = kernel_params[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            h_, gamma_, C_ = h, gammas[0], 100 
        best_C.append(C_)
        best_h.append(h_)
        best_gamma.append(gamma_)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not args.crossvalidation:
            break
    
    #---------------------------------
    # Printing and logging
    #---------------------------------
    if args.crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
                    np.mean(accuracy_scores) * 100,  
                    np.std(accuracy_scores) * 100))
    else:
        print('Final accuracy: {:2.3f} %'.format(np.mean(accuracy_scores)*100))

    # Save to file
    if args.crossvalidation or args.gridsearch:
        extension = ''
        if args.crossvalidation:
            extension += '_crossvalidation'
        if args.gridsearch:
            extension += '_gridsearch'
        results_filename = os.path.join(results_path, f'results_{dataset}'+extension+'.csv')
        n_splits = 10 if args.crossvalidation else 1
        pd.DataFrame(np.array([best_h, best_C, best_gamma, accuracy_scores]).T, 
                columns=[['h', 'C', 'gamma', 'accuracy']], 
                index=['fold_id{}'.format(i) for i in range(n_splits)]).to_csv(results_filename)
        print(f'Results saved in {results_filename}.')
    else:
        print('No results saved to file as --crossvalidation or --gridsearch were not selected.')

if __name__ == '__main__':
    main()