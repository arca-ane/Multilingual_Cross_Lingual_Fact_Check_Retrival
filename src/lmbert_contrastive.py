import os
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from utils import FactCheckDataset
from random import sample
from transformers import BertTokenizer, BertModel


class FactCheckModel:
    def __init__(self, dataset: FactCheckDataset, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initializes the class with the provided dataframes and loads the XLM-Roberta model.
        Args:
            dataset (FactCheckDataset): Dataset containing data from all the csv files
            hyperparameters: Not required for this model but for code compatibility
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fact_checks_df = dataset.training_data["fact_checks_df"]
        self.mapping_df = dataset.training_data["mapping_df"]
        self.posts_df = dataset.training_data["posts_df"]
        self.hyperparameters = hyperparameters
        self._set_default_hyperparameters()

        # Load mBERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.model.eval()  # Set the model to evaluation mode

        # Precompute embeddings for all fact-check claims
        self.fact_check_embeddings = self._compute_fact_check_embeddings(self.fact_checks_df['claim'])
        self.fact_checks = self.fact_checks_df[['fact_check_id', 'claim']].reset_index(drop=True)

    def _set_default_hyperparameters(self):
        default_hyperparameters = {
            'epochs': 5,         # Number of epochs
            'batch_size': 32,    # Batch size
            'margin': 1.0        # Contrastive Loss Margin
        }

        if self.hyperparameters is None:
            self.hyperparameters = {}
        for key, value in default_hyperparameters.items():
            self.hyperparameters.setdefault(key, value)

    def _compute_fact_check_embeddings(self, claims: List[str]) -> torch.Tensor:
        """
        Compute the embeddings for the given fact-check claims using mBERT with batching and GPU support.
        Args:
            claims (List[str]): List of fact-check claims.
        Returns:
            torch.Tensor: Tensor of embeddings for the fact-check claims.
        """

        # Move the model to GPU (if available)
        self.model.to(self.device)

        # Tokenize the claims in batches to improve performance
        batch_size = self.hyperparameters["batch_size"]
        embeddings = []

        # Process claims in batches
        for i in tqdm(range(0, len(claims), batch_size), desc="Processing batches"):
            batch_claims = claims[i:i + batch_size]
            batch_claims = list(batch_claims)  # Convert to list if necessary
            inputs = self.tokenizer(batch_claims, padding=True, truncation=True, return_tensors="pt").to(self.device)

            # Perform inference (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**inputs)

            claim_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, embedding_size]
            embeddings.append(claim_embeddings.cpu())  # Move embeddings to CPU to prevent GPU memory overflow

        # Concatenate all embeddings and return as a single tensor
        return torch.cat(embeddings, dim=0).to(self.device)


    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding for the input text using mBERT.
        Args:
            text (str): The text (social media post) for which to compute the embedding.
        Returns:
            torch.Tensor: The embedding for the input text.
        """

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        # Use the [CLS] token's embedding as the representation of the text
        return outputs.last_hidden_state[:, 0, :].to(self.device)

    def _contrastive_loss(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, label: int, margin: float = 1.0) -> torch.Tensor:
        """
        Compute the contrastive loss for a pair of embeddings.
        Args:
            embedding_a (torch.Tensor): Embedding of the first input.
            embedding_b (torch.Tensor): Embedding of the second input.
            label (int): 1 if the pair is similar, 0 otherwise.
            margin (float): Margin for dissimilar pairs.
        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        distance = torch.norm(embedding_a - embedding_b, p=2)
        loss = (label * distance ** 2) + ((1 - label) * torch.clamp(margin - distance, min=0) ** 2)
        return loss.mean()

    def predict(self, text: str, k: int = 1) -> List[Dict[str, str]]:
        """
        Predict the most relevant fact-checks for the given text (social media post) using contrastive distance.
        Args:
            text (str): The social media post to classify.
            k (int): The number of top fact-checks to retrieve.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the top k fact-checks' ids and claims.
        """

        # Get the embedding for the input text (social media post)
        self.model.eval()
        input_embedding = self._get_text_embedding(text)
        distances = torch.norm(self.fact_check_embeddings - input_embedding, dim=1)
        top_k_indices = distances.argsort()[:k]
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
        Train the model using contrastive loss.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.hyperparameters["epochs"]):
            total_loss = 0
            for _, row in tqdm(self.mapping_df.iterrows(), total=len(self.mapping_df), desc=f"Epoch {epoch + 1}"):
                # Retrieve the post and fact-check claim text for positive samples
                post_text_series = self.posts_df.loc[self.posts_df['post_id'] == row['post_id'], 'text']
                if post_text_series.empty:
                    continue
                post_text = post_text_series.values[0]
                claim_text = self.fact_checks_df.loc[self.fact_checks_df['fact_check_id'] == row['fact_check_id'], 'claim'].values[0]

                # Compute embeddings for positive samples
                embedding_post = self._get_text_embedding(str(post_text)).to(self.device)
                embedding_claim = self._get_text_embedding(str(claim_text)).to(self.device)
                loss = self._contrastive_loss(embedding_post, embedding_claim, 1, self.hyperparameters["margin"])
                total_loss += loss.item()
                negative_posts = sample(
                    [p for p in self.posts_df['post_id'].unique() if p != row['post_id']], 3
                )
                for neg_post_id in negative_posts:
                    neg_post_text = self.posts_df.loc[self.posts_df['post_id'] == neg_post_id, 'text'].values[0]
                    embedding_neg_post = self._get_text_embedding(str(neg_post_text)).to(self.device)
                    loss = self._contrastive_loss(embedding_neg_post, embedding_claim, 0, self.hyperparameters["margin"])
                    total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"\nEpoch {epoch + 1}, Loss: {total_loss / len(self.mapping_df)}")


    def save_model(self, model_dir: str = "lmbert_contrastive"):
        """
        Save model parameters for future use
        Args:
            model_dir (str) - Path where the parameters will be stored
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_params.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")


    def load_model(self, model_path: str = None):
        """
        Load model parameters for testing
        Args:
            model_path (str) - To name the directory where the parameters are stored
        """
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}")
