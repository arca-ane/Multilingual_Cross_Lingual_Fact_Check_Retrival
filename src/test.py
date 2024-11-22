import os
import ast
import git
import importlib
import argparse
from utils import Evaluator
from datetime import datetime

# Setup and parse command line arguments
def get_latest_date(directory, model_name):
    dates = []
    for subdir in os.listdir(directory):
        if subdir.startswith(model_name + "_"):
            date_str = subdir.split("_")[-1]
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            dates.append(date_obj)

    if dates:
        latest_date = max(dates)
        return latest_date.strftime("%Y%m%d")
    else:
        raise ValueError(f"No valid date directories found for model '{model_name}'.")

dir_path = os.path.dirname(os.path.realpath(__file__))
git_repo = git.Repo(dir_path, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
parser = argparse.ArgumentParser(description="Dynamically import a model class.")
parser.add_argument('--model', type=str, required=True, help="Name of the model file to import (e.g., 'lmbert', 'bert').")
parser.add_argument('--use_actual', action='store_false', help="Use trial data (default: True)")
parser.add_argument('--date', type=str, help="Date in yyyymmdd format (default: latest available date).")
parser.add_argument('--ratio', type=float, default=1.0, help="Ratio of the amount of data to use (default: 1.0)")
args = parser.parse_args()
if args.date is None:
    args.date = get_latest_date(f"{git_root}/trained_models", args.model)


# Initialize the evaluator and load the split data
if args.use_actual:
    evaluator = Evaluator(
        git_root + "/data/trial_fact_checks.csv",
        git_root + "/data/trial_posts.csv",
        git_root + "/data/trial_data_mapping.csv",
        args.ratio
    )
else:
    evaluator = Evaluator(
        git_root + "/data/fact_checks.csv",
        git_root + "/data/posts.csv",
        git_root + "/data/fact_check_post_mapping.csv",
        args.ratio
    )

# Specify the path to the saved model parameters
model_path = git_root + f"/trained_models/{args.model}_{args.date}/model_params.pt"
print("Using model from ", model_path, "for testing")

# Load the model
print("INITIALISING MODEL")
model_module = importlib.import_module(args.model)
FactCheckModel = getattr(model_module, "FactCheckModel")
model = FactCheckModel(evaluator.dataset)
print("Successfully loaded FactCheckModel from ", args.model + ".py")

# Initialize the model and load parameters
print("LOADING PRE-TRAINED MODEL PARAMETERS")
model.load_model(model_path)

# Evaluate Success@K and MRR for the model
print("STARTING MODEL EVALUATION for k=3")
evaluation_results = evaluator.evaluate(model, k=3)
print(evaluation_results)
print("STARTING MODEL EVALUATION for k=5")
evaluation_results = evaluator.evaluate(model, k=5)
print(evaluation_results)
print("STARTING MODEL EVALUATION for k=10")
evaluation_results = evaluator.evaluate(model, k=10)
print(evaluation_results)
print("FINISHED MODEL EVALUATION")
