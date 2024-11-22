from preoprocess import Preprocessor
from model1 import FactCheckModel

# Initialize and load data
preprocessor = Preprocessor("data/trial_data_mapping.csv", "data/trial_fact_checks.csv", "data/trial_posts.csv")
preprocessor.load_data()
training_data = preprocessor.prepare_data()

# Define training hyperparameters (example)
hyperparameters = {
    "learning_rate": 0.001,
    "epochs": 10
}

# Initialize and train the model
model = FactCheckModel(training_data=training_data, hyperparameters=hyperparameters)
model.train()

# Save trained model parameters
model.save_model(model_name="lmbert")