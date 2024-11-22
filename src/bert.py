import os
import torch
from tqdm import tqdm
from typing import List, Dict
from utils import FactCheckDataset
from transformers import BertTokenizer, BertModel


class FactCheckModel:
    def __init__(self, dataset: FactCheckDataset, hyperparameters = None):
        """
        Initializes the class with the provided dataframes and loads the XLM-Roberta model.
        Args:
            dataset (FactCheckDataset): Dataset containing data from all the csv files
            hyperparameters: Not required for this model but for code compatibility
        """
        self.fact_checks_df = dataset.training_data["fact_checks_df"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained tokenizer and model (e.g., BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.fact_check_embeddings = self._compute_fact_check_embeddings(self.fact_checks_df['claim'].tolist())
        self.fact_checks = self.fact_checks_df[['fact_check_id', 'claim']].reset_index(drop=True)


    def _compute_fact_check_embeddings(self, claims: List[str]) -> torch.Tensor:
        """
        Compute the embeddings for the given fact-check claims using a BERT-based tokenizer and model.
        Args:
            claims (List[str]): List of fact-check claims.
        Returns:
            torch.Tensor: Tensor of embeddings for the fact-check claims.
        """

        embeddings = []
        # Tokenize the claims in batches to improve speed (and use GPU if available)
        batch_size = 32  # Adjust batch size based on available memory
        for i in tqdm(range(0, len(claims), batch_size)):
            batch_claims = claims[i:i+batch_size]
            inputs = self.tokenizer(batch_claims, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            claim_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(claim_embedding)

        # Stack all the embeddings and return as a tensor
        return torch.vstack(embeddings).to(self.device)


    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding for the input text using mBERT.
        Args:
            text (str): The text (social media post) for which to compute the embedding.
        Returns:
            torch.Tensor: The embedding for the input text.
        """

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token's embedding as the representation of the text
        return outputs.last_hidden_state[:, 0, :].to(self.device)


    def predict(self, text: str, k: int = 1) -> List[Dict[str, str]]:
        """
        Predict the most relevant fact-checks for the given text (social media post).
        Args:
            text (str): The social media post to classify.
            k (int): The number of top fact-checks to retrieve.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the top k fact-checks' ids and claims.
        """

        # Get the embedding for the input text (social media post)
        input_embedding = self._get_text_embedding(text)
        similarities = torch.cosine_similarity(input_embedding, self.fact_check_embeddings).to(self.device)
        top_k_indices = similarities.argsort(descending=True)[:k]
        top_k_indices = top_k_indices.cpu().numpy().tolist()

        # Prepare the results: fact-check IDs and claims
        results = []
        for idx in top_k_indices:
            fact_check_id = self.fact_checks.loc[idx, 'fact_check_id']
            claim = self.fact_checks.loc[idx, 'claim']
            results.append({"fact_check_id": fact_check_id, "claim": claim})

        return results
    
    def train(self):
        """
        Train the model as per requirements
        """

        print("Training not required for this model, can directly test.")


    def save_model(self, model_dir: str = "lmbert"):
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
