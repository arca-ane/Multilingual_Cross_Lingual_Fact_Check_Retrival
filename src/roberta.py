import os
import torch
import pandas as pd
from typing import List, Dict
from utils import FactCheckDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class FactCheckModel:
    def __init__(self, dataset: FactCheckDataset, hyperparameters = None):
        """
        Initializes the class with the provided dataframes and loads the XLM-Roberta model.
        Args:
            dataset (FactCheckDataset): Dataset containing data from all the csv files
            hyperparameters: Not required for this model but for code compatibility
        """
        self.fact_checks_df = dataset.training_data["fact_checks_df"]

        # Load the tokenizer and model for XLM-Roberta
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.model = AutoModel.from_pretrained('xlm-roberta-base')

        # Precompute embeddings for all fact-check claims
        self.fact_check_embeddings = self._compute_fact_check_embeddings(self.fact_checks_df['claim'])


    def _compute_fact_check_embeddings(self, claims: List[str]) -> torch.Tensor:
        """
        Computes the embeddings for all fact-check claims.
        Args:
            claims (List[str]): List of fact-check claims.
        Returns:
            torch.Tensor: Tensor of embeddings for the fact-check claims.
        """
        # Move the model to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        claims = claims.tolist() if isinstance(claims, pd.Series) else claims
        inputs = self.tokenizer(claims, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings


    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Generates the embedding for the input text.
        Args:
            text (str): The text for which to generate an embedding.
        Returns:
            torch.Tensor: The embedding for the input text.
        """
        # Move the model to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embedding


    def predict(self, text: str, k: int = 1) -> List[Dict[str, str]]:
        """
        Predicts the top `k` most relevant fact-check claims for the given input text.
        Args:
            text (str): Input text to find the closest fact-check claims.
            k (int, optional): Number of closest fact-check claims to return. Defaults to 1.
        Returns:
            List[Dict[str, str]]: List of dictionaries containing 'fact_check_id' and 'claim' for the top `k` fact-checks.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_embedding = self._get_text_embedding(text)
        input_embedding_cpu = input_embedding.cpu().numpy()
        fact_check_embeddings_cpu = self.fact_check_embeddings.cpu().numpy()
        similarities = cosine_similarity(input_embedding_cpu, fact_check_embeddings_cpu)
        top_k_indices = similarities.argsort()[0][-k:][::-1]  # Top k indices sorted by similarity

        # Prepare the result as a list of dictionaries with 'fact_check_id' and 'claim'
        top_k_results = []
        for idx in top_k_indices:
            fact_check_id = self.fact_checks_df.iloc[idx]['fact_check_id']
            claim = self.fact_checks_df.iloc[idx]['claim']
            top_k_results.append({"fact_check_id": fact_check_id, "claim": claim})

        return top_k_results


    def train(self):
        """
        Train the model as per requirements
        """

        print("Training not required for this model, can directly test.")


    def save_model(self, model_dir: str = "roberta"):
        """
        Save model parameters for future use
        Args:
            model_dir (str) - Path where the parameters will be stored
        """

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_params.pt")
        with open(model_path, 'w') as file:
            file.write("Non parameterized model")  # Saving model parameters
        print(f"Model saved at {model_path}")


    def load_model(self, model_path: str = None):
        """
        Load model parameters for testing
        Args:
            model_path (str) - To name the directory where the parameters are stored
        """

        pass
