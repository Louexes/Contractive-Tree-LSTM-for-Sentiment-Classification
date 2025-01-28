import argparse
import matplotlib.pyplot as plt
import pandas as pd


PLOT_BASE_PATH = "artifacts/plots/"
RESULT_BASE_PATH = "artifacts/results/"


def calculate_aggregates(results_file: str):
    """
    Reads a results CSV file, calculates the mean and standard deviation of test accuracies
    grouped by model name, and saves the aggregated results to another CSV file.

    Args:
        results_file (str): Path to the input CSV file containing experiment results.
        output_file (str): Path to save the aggregated results as a CSV file.
    """
    results_file = RESULT_BASE_PATH + results_file
    output_file = results_file.replace(".csv", "_agg.csv")
    df = pd.read_csv(results_file)
    df['best_test_accuracy'] = pd.to_numeric(df['best_test_accuracy'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    
    aggregates = df.groupby('model_name').agg(
        mean_test_accuracy=('best_test_accuracy', 'mean'),
        std_test_accuracy=('best_test_accuracy', 'std'),
        mean_runtime=('runtime', 'mean'),
        std_runtime=('runtime', 'std')
    ).reset_index()
    aggregates.to_csv(output_file, index=False)


def plot_results(results_file_name: str, model_name: str, name: str = "results_plot", title: str = "Accuracy and Loss") -> None:
    """
    Plot the loss and accuracy values for a given model and save the plot.

    Args:
        results_file_name (str): Name of the results CSV file.
        model_name (str): Name of the model to filter results.
        name (str): Name of the output plot file.
        title (str): Title of the plot.
    """
    results_file = RESULT_BASE_PATH + results_file_name
    df = pd.read_csv(results_file)

    model_data = df[df['model_name'] == model_name]

    if model_data.empty:
        raise ValueError(f"No data found for model name: {model_name}")

    # Extract the first row's loss and accuracy values
    losses = eval(model_data.iloc[0]['losses'])  # Convert string to list
    accuracies = eval(model_data.iloc[0]['accuracies'])

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Evaluation cycle')
    ax1.set_ylabel('Loss')
    ax1.plot(losses, color='blue', label='Loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(accuracies, color='red', label='Accuracy')

    plt.title(title)
    ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(f"{PLOT_BASE_PATH}{name}.jpg", dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment results and plots.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Subparser for calculate_aggregates
    parser_agg = subparsers.add_parser("calculate_aggregates", help="Calculate aggregates from results.")
    parser_agg.add_argument("results_file", type=str, help="Name of the results CSV file.")

    # Subparser for plot_results
    parser_plot = subparsers.add_parser("plot_results", help="Plot training results.")
    parser_plot.add_argument("--results_file_name", type=str, default="main_comparison.csv", help="Name of the results file")
    parser_plot.add_argument("--model_name", type=str, default="BOW", help="Name of the model")
    parser_plot.add_argument("--name", type=str, default="results_plot", help="Name for the output plot file.")
    parser_plot.add_argument("--title", type=str, default="Accuracy and Loss", help="Title of the plot.")

    args = parser.parse_args()

    if args.command == "calculate_aggregates":
        calculate_aggregates(args.results_file)

    elif args.command == "plot_results":
        plot_results(results_file_name=args.results_file_name, model_name=args.model_name, name=args.name, title=args.title)
