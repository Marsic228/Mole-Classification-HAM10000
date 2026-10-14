# experiment_run.py
# Goal: ExperimentRun class that logs one training experiment
# You need: datetime, json, os

import json
import os
from datetime import datetime


class ExperimentRun:
    def __init__(self, experiment_id, model_name, dataset_config_path,
                 num_epochs, learning_rate, notes=""):
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.dataset_config_path = dataset_config_path
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.notes = notes
        self.created_at = datetime.now().isoformat()
        self.metrics = {}
        self.status = "created"

    def set_metrics(self, accuracy, loss, extra=None):
        self.metrics = {"accuracy" : accuracy, "loss": loss}
        if extra is not None:
            self.metrics.update(extra)
        self.status = "completed"

    def to_dict(self):
        return {
            "experiment_id" : self.experiment_id,
            "model_name" : self.model_name,
            "dataset_config_path" : self.dataset_config_path,
            "num_epochs" : self.num_epochs,
            "learning_rate" : self.learning_rate,
            "notes" : self.notes,
            "created_at" : self.created_at,
            "metrics" : self.metrics,
            "status" : self.status
        }
    

    @classmethod
    def from_dict(cls, data):
        run = cls(
            experiment_id=data.get("experiment_id", ""),
            model_name=data.get("model_name", ""),
            dataset_config_path=data.get("dataset_config_path", ""),
            num_epochs=float(data.get("num_epochs", 0)),
            learning_rate=float(data.get("learning_rate", 0.0)),    
            notes=data.get("notes", ""),
        )
        run.created_at=data.get("created_at","")
        run.metrics = data.get("metrics", {})
        run.status=data.get("status", "")
        return run

    def __repr__(self):
        return f"ExperimentRun(id= {self.experiment_id}, model= {self.model_name}, status= {self.status})"



def save_experiment(experiment, path):
    with open(path, "w") as file:
        json.dump(experiment.to_dict(), file, indent=2)


def load_experiment(path):
    with open(path, "r") as file:
        loaded = json.load(file)
    return ExperimentRun.from_dict(loaded)

if __name__ == "__main__":
    experiment = ExperimentRun(
        experiment_id = "0253095",
        model_name = "gpt",
        dataset_config_path = "data/HAM10000_metadata.csv",
        num_epochs = "3",
        learning_rate = "2"
    )
    print(experiment)
    experiment.set_metrics(0.85, 0.42)
    print(experiment)
    path = "test_experiment.json"
    save_experiment(experiment, path)
    loaded = load_experiment(path)
    print(loaded)