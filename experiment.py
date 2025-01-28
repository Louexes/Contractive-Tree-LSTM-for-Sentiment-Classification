# NOTE: This module does not contain the experiments. It only contains a helper class that records the performance.

import os
import csv
import time
from datetime import datetime


class Experiment:
    # Class-level variable to keep track of the experiment count across runtimes
    _experiment_count = 0

    def __init__(self, model_name, model):
        # Increment the experiment count to generate a unique ID
        self.ID = Experiment._experiment_count
        Experiment._experiment_count += 1

        self.losses = []
        self.accuracies = []
        self.best_train_acc = None
        self.best_dev_acc = None
        self.best_test_acc = None
        self.model_name = model_name
        self.model = model
        self.start_time = time.time()

    def flatten(self):
        # Combine all necessary experiment information into a flattened dictionary
        flattened_dict = {
            "ID": self.ID,
            "losses": str(self.losses),
            "accuracies": str(self.accuracies),
            "best_train_accuracy": self.best_train_acc,
            "best_dev_accuracy": self.best_dev_acc,
            "best_test_accuracy": self.best_test_acc,
            "model_name": self.model_name,
            "model": str(self.model),
            "runtime": time.time() - self.start_time,
            "PID": os.getpid(),
            "timestamp": datetime.now(),
        }
        return flattened_dict

    def store(self, csv_file):
        # Append the flattened experiment data to a CSV file
        flattened_data = self.flatten()
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=flattened_data.keys())
            if not file_exists:
                writer.writeheader()  # Write header if the file is empty
            writer.writerow(flattened_data)
