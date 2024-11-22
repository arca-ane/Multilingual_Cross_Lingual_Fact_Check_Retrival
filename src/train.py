import os
import ast
import git
import argparse
import importlib
from datetime import datetime
from utils import Preprocessor

# Setup and parse command line arguments
parser = argparse.ArgumentParser(description="Dynamically import a model class.")
parser.add_argument('--model', type=str, required=True, help="Name of the model file to import (e.g., 'lmbert', 'bert').")
parser.add_argument('--use_actual', action='store_false', help="Use trial data (default: True)")
parser.add_argument('--ratio', type=float, default=1.0, help="Ratio of the amount of data to use (default: 1.0)")
parser.add_argument('--hyperparameters', type=ast.literal_eval, default=None, help="Hyperparameter dictionary (default: Model specific)")
args = parser.parse_args()

# Initialize and load data
dir_path = os.path.dirname(os.path.realpath(__file__))
git_repo = git.Repo(dir_path, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

if args.use_actual:
    preprocessor = Preprocessor(
        git_root + "/data/trial_fact_checks.csv",
        git_root + "/data/trial_posts.csv",
        git_root + "/data/trial_data_mapping.csv"
    )
else:
    preprocessor = Preprocessor(
        git_root + "/data/fact_checks.csv",
        git_root + "/data/posts.csv",
        git_root + "/data/fact_check_post_mapping.csv",
    )

dataset = preprocessor.prepare_data(args.ratio)
hyperparameters = args.hyperparameters

# Load the model
print("INITIALISING MODEL")
model_module = importlib.import_module(args.model)
FactCheckModel = getattr(model_module, "FactCheckModel")
model = FactCheckModel(dataset, hyperparameters)
print("Successfully loaded FactCheckModel from ", args.model + ".py")

# Train the model
print("STARTING MODEL TRAINING")
model.train()

# Save trained model parameters
print("STORING LEARNED PARAMETERS")
date_str = datetime.now().strftime("%Y%m%d")
model_path = git_root + f"/trained_models/{args.model}_{date_str}"
model.save_model(model_path)
