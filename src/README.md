# WWL Package

## Installation
To install `wwl`, run the following:
```sh
$ pip install cython numpy
$ pip install wwl
```

## Usage
WWL can be used to compute the pairwise kernel matrix between a list of Graphs.
The kernel function `wwl` takes as input a list of igraph `Graph` objects. It can also take their node features (if they are continuously attributed), the number of iterations for the embedding scheme,
the value for gamma in the Laplacian kernel, and a flag for sinkhorn approximations.
```python
from wwl import wwl

# load the graphs
graphs = [ig.read(fname) for fname in graph_filenames]

# load node features for continuous graphs
node_features = np.load(path_to_node_features)

# compute the kernel
kernel_matrix = wwl(graphs, node_features=node_features, num_iterations=4)

# use in SVM
from sklearn.svm import SVC

train_index, test_index = np.load(train_index_path), np.load(test_index_path)
y = np.load(path_to_labels)
K_train = kernel_matrix[train_index][:,train_index]
K_test = kernel_matrix[test_index][:,train_index]

svm = SVC(kernel='precomputed') # For a Krein SVM, please refer to krein.py
svm.fit(K_train)

y_predict = svm.predict(K_test)
```

Please see `utilities.wwl_custom_grid_search_cv` for a custom crossvalidation to cross-validate the number of iterations, gammas in the Laplacian kernel, and other parameters for the SVM.  
