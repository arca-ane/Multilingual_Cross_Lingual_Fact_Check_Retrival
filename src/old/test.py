from preoprocess import Preprocessor
from model1 import FactCheckModel

# Initialize the preprocessor and load the split data
preprocessor = Preprocessor("trial_data.csv")
(X_train, y_train), (X_test, y_test) = preprocessor.load_and_split_data()

# Specify the path to the saved model parameters
model_path = "trained_model/lmbert_20241029/model_params.pt"

# Initialize the model and load parameters
model = FactCheckModel()
model.load_model(model_path)

# Perform predictions on the test set
for post, label in zip(X_test, y_test):
    fact_check_result = model.predict(post)
    print(f"Post: {post}\nPredicted Fact-check: {fact_check_result}\nTrue Label: {label}\n")
