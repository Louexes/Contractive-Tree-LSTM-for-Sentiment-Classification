import numpy as np
import argparse
import logging
import inspect
import os
from collections import OrderedDict
from models import *
from data_processing.utils import get_datasets, bin_sentences, prepare_example_bow, prepare_example_lstm, prepare_example_tree_lstm, evaluate_model_on_bins, Vocabulary
from experiment import Experiment
from torch import optim
from data_processing.utils import (
    prepare_treelstm_minibatch,
    get_minibatch,
    prepare_example,
    get_examples,
)
from evaluate import evaluate, simple_evaluate
from training import train_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os


# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(process)d - %(levelname)s - %(message)s"
)


MODELS = [BOW, CBOW, DeepCBOW, LSTMClassifier, TreeLSTMClassifier]
MODEL_NAMES = {model: model.__name__ for model in MODELS}
BENCHMARKS = [
    "main_comparison",
    "tree_lstm_comparison",
    "cae_tree_lstm_comparison",
    "sentence_length_comparison",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sentence_length_comparison(v: Vocabulary, t2i: OrderedDict, vectors: np.ndarray, num_trainings: int = 3) -> None:
    "Run inference for the performance comparison for different sentence length for all models from the notebook."

    # Define bins
    bins = [0, 5, 10, 15, 20, 25, 30, np.inf]
    bin_labels = [
        f'{int(bins[i]+1)}-{int(bins[i+1])}' if bins[i+1] != np.inf else f'>{int(bins[i])}'
        for i in range(len(bins)-1)
    ]

    # Bin the test data
    binned_data = bin_sentences(get_datasets()[2], bins)

    # Initialize a DataFrame to store the performance of each model for different seeds
    performance_df = pd.DataFrame(columns=['Model', 'Seed', 'Bin', 'Accuracy'])

    # Function to evaluate and store performance
    def evaluate_and_store_performance(model, model_name, seed, binned_data, prep_fn):
        performances = evaluate_model_on_bins(model, binned_data, prep_fn, bins=bins)
        for bin_label, performance in zip(bin_labels, performances):
            performance_df.loc[len(performance_df)] = [model_name, seed, bin_label, performance]

    # Evaluate and store performance for each model and seed
    for seed in range(3):
        bow_model = BOW(len(v.w2i), len(t2i), vocab=v)
        checkpoint = torch.load(f'artifacts/checkpoints/main_comparison/BOW_{seed}.pt', weights_only=False)
        bow_model.load_state_dict(checkpoint['state_dict'])

        cbow_model = CBOW(len(v.w2i), 300, len(t2i), vocab=v)
        checkpoint = torch.load(f'artifacts/checkpoints/main_comparison/CBOW_{seed}.pt', weights_only=False)
        cbow_model.load_state_dict(checkpoint['state_dict'])

        deepcbow_model = DeepCBOW(len(v.w2i), 300, len(t2i), vocab=v)
        checkpoint = torch.load(f'artifacts/checkpoints/main_comparison/DeepCBOW_{seed}.pt', weights_only=False)
        deepcbow_model.load_state_dict(checkpoint['state_dict'])

        lstm_model = LSTMClassifier(len(v.w2i), 300, 168, len(t2i), v)
        checkpoint = torch.load(f'artifacts/checkpoints/main_comparison/LSTMClassifier_{seed}.pt', weights_only=False)
        lstm_model.load_state_dict(checkpoint['state_dict'])

        tree_model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v)
        checkpoint = torch.load(f'artifacts/checkpoints/main_comparison/TreeLSTMClassifier_{seed}.pt', weights_only=False)
        tree_model.load_state_dict(checkpoint['state_dict'])

        models_and_prep_fns = [
            {'model': bow_model, 'name': 'BOW', 'prep_fn': prepare_example_bow},
            {'model': cbow_model, 'name': 'CBOW', 'prep_fn': prepare_example_bow},
            {'model': deepcbow_model, 'name': 'Deep CBOW', 'prep_fn': prepare_example_bow},
            {'model': lstm_model, 'name': 'LSTM', 'prep_fn': prepare_example_lstm},
            {'model': tree_model, 'name': 'Tree LSTM', 'prep_fn': prepare_example_tree_lstm},
        ]

        for item in models_and_prep_fns:
            evaluate_and_store_performance(item['model'], item['name'], seed, binned_data, item['prep_fn'])

    # Save the DataFrame to a CSV file
    performance_df.to_csv('artifacts/results/sequence_length_results.csv', index=False)

    # Calculate average performance and standard deviation for each model over all seeds
    avg_performance_df = performance_df.groupby(['Model', 'Bin']).agg({'Accuracy': ['mean', 'std']}).reset_index()
    avg_performance_df.columns = ['Model', 'Bin', 'Mean_Accuracy', 'Std_Accuracy']

    # Plotting
    plt.figure(figsize=(12, 8))
    matplotlib.rcParams.update({'font.size': 20})

    for model_name in avg_performance_df['Model'].unique():
        model_data = avg_performance_df[avg_performance_df['Model'] == model_name]
        plt.errorbar(model_data['Bin'], model_data['Mean_Accuracy'], yerr=model_data['Std_Accuracy'], marker='o', label=model_name, capsize=5)

    plt.xlabel('Sentence Length Bin')
    plt.ylabel('Average Accuracy')
    plt.title('Average Model Performance vs. Sentence Length')
    plt.legend(
        ncol=5, 
        columnspacing=0.5,
        handletextpad=0.3
    )
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./artifacts/plots/sentence_length_comparison.png", dpi=100)
    plt.show()

    matplotlib.rcParams.update({'font.size': 10}) # Resetting back to the default to not interfere with other functions.


def cae_tree_lstm_comparison(v: Vocabulary, t2i: OrderedDict, vectors: np.ndarray, num_trainings: int = 3) -> None:
    "Compare our CAE TreeLSTM variant to the classical TreeLSTM."

    data = get_datasets()
    experiment_name = inspect.stack()[0][3]

    for use_cae in [True, False]:
        for seed in range(num_trainings):
            torch.manual_seed(seed)

            if use_cae == False:
                model_path = f'models/tree_lstm_model_{seed}.pt'
                if os.path.exists(model_path):
                    model = torch.load(model_path, map_location=DEVICE)
                else:
                    model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v)
                    with torch.no_grad():
                        model.embed.weight.data.copy_(torch.from_numpy(vectors))
                        model.embed.weight.requires_grad = False

                    model = model.to(DEVICE)
                    model.seed = seed

                    name_prefix = "CAE" if use_cae else ""
                    experiment = Experiment(name_prefix + model.__class__.__name__, str(model))
                    experiment.name = experiment_name

                    optimizer = optim.Adam(model.parameters(), lr=2e-4)
                    results = train_model(
                        model,
                        optimizer,
                        data=data,
                        num_iterations=30000,
                        print_every=250,
                        eval_every=250,
                        prep_fn=prepare_treelstm_minibatch,
                        eval_fn=evaluate,
                        batch_fn=get_minibatch,
                        batch_size=25,
                        eval_batch_size=25,
                        experiment=experiment,
                    )
                    experiment.store(csv_file=f"artifacts/results/{experiment_name}.csv")
            else:
                model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v, use_cae=use_cae)
                with torch.no_grad():
                    model.embed.weight.data.copy_(torch.from_numpy(vectors))
                    model.embed.weight.requires_grad = False

                model = model.to(DEVICE)
                model.seed = seed

                name_prefix = "CAE" if use_cae else ""
                experiment = Experiment(name_prefix + model.__class__.__name__, str(model))
                experiment.name = experiment_name

                optimizer = optim.Adam(model.parameters(), lr=2e-4)
                results = train_model(
                    model,
                    optimizer,
                    data=data,
                    num_iterations=30000,
                    print_every=250,
                    eval_every=250,
                    prep_fn=prepare_treelstm_minibatch,
                    eval_fn=evaluate,
                    batch_fn=get_minibatch,
                    batch_size=25,
                    eval_batch_size=25,
                    experiment=experiment,
                )
                experiment.store(csv_file=f"artifacts/results/{experiment_name}.csv")


def main_comparison(v: Vocabulary, t2i: OrderedDict, vectors: np.ndarray, num_trainings: int = 3) -> None:
    "Comparing all models to answer mandatory questions 1 and 2."

    optimizer_lr = {
        "BOW": 0.0005,
        "CBOW": 0.0005,
        "DeepCBOW": 0.0005,
        "LSTMClassifier": 2e-4,
        "TreeLSTMClassifier": 2e-4,
    }

    num_iterations = {
        "BOW": 60000,
        "CBOW": 10000,
        "DeepCBOW": 10000,
        "LSTMClassifier": 30000,
        "TreeLSTMClassifier": 30000,
    }

    data = get_datasets()
    experiment_name = inspect.stack()[0][3]

    for model_class in MODELS:
        for seed in range(num_trainings):

            torch.manual_seed(seed)

            if model_class == BOW:
                model = model_class(len(v.w2i), len(t2i), vocab=v)
            elif model_class == CBOW:
                model = model_class(len(v.w2i), 300, len(t2i), vocab=v)
            elif model_class == DeepCBOW:
                model = model_class(len(v.w2i), 300, len(t2i), vocab=v)
                model.embed.weight.data.copy_(torch.from_numpy(vectors))
            elif model_class == LSTMClassifier:
                model = model_class(len(v.w2i), 300, 168, len(t2i), v)
            elif model_class == TreeLSTMClassifier:
                model = model_class(len(v.w2i), 300, 150, len(t2i), v)

            batch_size = 1

            if "LSTM" in model_class.__name__:
                with torch.no_grad():
                    model.embed.weight.data.copy_(torch.from_numpy(vectors))
                    model.embed.weight.requires_grad = False
                batch_size = 25

            model = model.to(DEVICE)
            model.seed = seed
            # Define experiment for performance recording
            experiment = Experiment(model.__class__.__name__, str(model))
            experiment.name = experiment_name

            optimizer = optim.Adam(model.parameters(), lr=optimizer_lr[model.__class__.__name__])

            results = train_model(
                model,
                optimizer,
                data,
                num_iterations=num_iterations[model.__class__.__name__],
                print_every=1000,
                eval_every=1000,
                batch_fn=get_minibatch if model.__class__ == TreeLSTMClassifier else get_examples,
                prep_fn=(
                    prepare_treelstm_minibatch
                    if model.__class__ == TreeLSTMClassifier
                    else prepare_example
                ),
                eval_fn=(
                    evaluate if model.__class__ == TreeLSTMClassifier else simple_evaluate
                ),
                batch_size=batch_size,
                eval_batch_size=None,
                experiment=experiment,
            )
            experiment.store(csv_file=f"artifacts/results/{experiment_name}.csv")


def tree_lstm_comparison(v: Vocabulary, t2i: OrderedDict, vectors: np.ndarray, num_trainings: int = 3) -> None:
    "Comparing TreeLSTM trained on the original dataset vs. the dataset containing all sub-trees."

    experiment_name = inspect.stack()[0][3]

    for use_expanded_data in [True, False]:
        data = get_datasets(use_expanded_data=use_expanded_data)
        for seed in range(num_trainings):
            torch.manual_seed(seed)

            # Define model
            model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v)

            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(vectors))
                model.embed.weight.requires_grad = False

            model = model.to(DEVICE)
            model.seed = seed

            # Define experiment for performance recording
            name_prefix = "SubTree" if use_expanded_data else ""
            experiment = Experiment(name_prefix + TreeLSTMClassifier.__name__, str(model))
            experiment.name = experiment_name

            optimizer = optim.Adam(model.parameters(), lr=2e-4)
            results = train_model(
                model,
                optimizer,
                data=data,
                num_iterations=30000,
                print_every=250,
                eval_every=250,
                prep_fn=prepare_treelstm_minibatch,
                eval_fn=evaluate,
                batch_fn=get_minibatch,
                batch_size=25,
                eval_batch_size=25,
                experiment=experiment,
            )
            experiment.store(csv_file=f"artifacts/results/{experiment_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="trees/",
        help="Base path for input and output files. Default is 'trees/'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=BENCHMARKS,
        required=True,
        help="Mode to run the script. Choices are: 'main_comparison', 'tree_lstm_comparison', 'cae_tree_lstm_comparison', 'sentence_length_comparison'.",
    )
    parser.add_argument(
        "--num_trainings",
        type=int,
        default=3,
        help="Number of training runs for each model."
    )

    args = parser.parse_args()
    mode = args.mode
    if args.mode not in BENCHMARKS:
        raise ValueError(
            f"Invalid mode: {args.mode}. Choose one of the allowed modes: {BENCHMARKS}"
        )

    EXPERIMENTS = [
        main_comparison,
        tree_lstm_comparison,
        cae_tree_lstm_comparison,
        sentence_length_comparison,
    ]
    EXPERIMENT_NAME_TO_FUNC = {
        experiment_func.__name__: experiment_func for experiment_func in EXPERIMENTS
    }

    # Define BOW classes
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

    # Load vocabulary
    glove_file = "glove.840B.300d.sst.txt"
    word_set = set()

    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split()[0]
            word_set.add(word)

    v = Vocabulary()
    vectors = []

    # Add <unk> and <pad> tokens
    v.add_token("<unk>")
    v.add_token("<pad>")
    np.random.seed(0) # Fix see here for reproducable results.
    vectors.append(np.random.randn(300))  # Random vector for <unk>
    vectors.append(np.zeros(300))  # Zero vector for <pad>

    # Read the GloVe file and add words to the vocabulary and vectors list
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word_set:
                vector = np.array(parts[1:], dtype=np.float32)
                v.add_token(word)
                vectors.append(vector)

    # Convert vectors list to a numpy array
    vectors = np.stack(vectors, axis=0)

    ### Run experiment function
    exp_func = EXPERIMENT_NAME_TO_FUNC[mode]
    exp_func(v, t2i, vectors, num_trainings=args.num_trainings)

    logging.info(f"Completed running experiment: {mode}")
