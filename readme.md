# Contractive Tree LSTM for Sentiment Classification
This readme contains all necessary instructions on how our results reported as part of the "Further Research" section can be reproduced.

First, install pandas.
```
pip install pandas
```
NOTE: We are not including model checkpoints here to avoid inflating the size of the submission. Please feel free to e-mail us in case you would like to have them for the evaluation and we will make sure you'll have them as soon as possible.

### Dataset Expansion
As of the "Further Research" section, we are supposed to create a dataset that contains all sub-trees for each tree in the SST dataset. To perform this, run the following in your command line (from the main directory of the repository):
```
python -m data_processing.dataset_expansion
```


## Run experiments
The main comparison comparing all the models (except for the CTreeLSTM) is run using:
```
python -m benchmark --mode main_comparison
```
All checkpoints, results, aggregated results and plots can be found in the artifacts directory.
Experiments are run across three different PyTorch seeds, the numpy seed is fixed to 0.

### Standard TreeLSTM vs. CTreeLSTM
This experiment compares our TreeLSTM variant that uses a contractive regularization term instead of a dropout.
```
python -m benchmark --mode cae_tree_lstm_comparison
```

### TreeLSTM with and without SubTrees
In order to run this, make sure to run the "Dataset Expansion" first as described above.
```
python -m benchmark --mode tree_lstm_comparison
```


### Sensitivity to sentence length
This experiment makes use of the checkpoints from the main comparison. In order to run this, make sure there are three checkpoints for each model in artifacts/checkpoints/main_comparison. If that is not the case, run the experiment for the main comparison first.
```
python -m benchmark --mode sentence_length_comparison
```


## Get aggregated results from raw results
Replace "main_comparison" by the name of the experiment that the aggregates should be generated for.
```
python -m plots_and_tables calculate_aggregates main_comparison.csv

