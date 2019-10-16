# Experiment script

This script reproduces the experiment presented in the paper for 2 datasets.

## Dependencies
Before launching the script, ensure to ahve the following dependencies installed:
- `numpy`
- `scikit-learn`
- `POT` ([github](https://github.com/rflamary/POT))
- `python-igraph`

## Run the experiments
In its simplest instance, you can run the example code as follows:
```bash
python3 main.py --dataset MUTAG # or ENZYMES
```

Additionally, you can play with the following options (run `python3 main.py -h` to display them):
```bash
usage: main.py [-h] [-d {MUTAG,ENZYMES}] [--crossvalidation] [--gridsearch]
               [--sinkhorn] [--h H]

optional arguments:
  -h, --help            show this help message and exit
  -d {MUTAG,ENZYMES}, --dataset {MUTAG,ENZYMES}
                        Provide the dataset name (MUTAG or Enzymes)
  --crossvalidation     Enables a 10-fold crossvalidation
  --gridsearch          Enable grid search
  --sinkhorn            Use sinkhorn approximation
  --h H                 (Max) number of WL iterations


# For example:
python main.py -d ENZYMES --gridsearch --crossvalidation --h 7
```