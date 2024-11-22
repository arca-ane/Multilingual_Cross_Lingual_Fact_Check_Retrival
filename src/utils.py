import pandas as pd
from tqdm import tqdm

class FactCheckDataset:
    def __init__(self, fact_checks_df: pd.DataFrame, posts_df: pd.DataFrame,
                 mapping_df: pd.DataFrame, ratio: float = 1.0):
        self.mapping_df = mapping_df
        self.fact_checks_df = fact_checks_df
        self.posts_df = posts_df
        self.ratio = ratio

        # Setup dataset partitions as per requirement
        self._setup_dataset()

    def _setup_dataset(self):
        self._filter_data()

        # Setup important dataset parameters
        self.training_data = {"fact_checks_df": self.fact_checks_df_filtered,
                              "posts_df": self.posts_df_filtered,
                              "mapping_df": self.mapping_df_filtered}

        self.test_data = None
    
    def _filter_data(self):
        num : int = int(self.ratio * len(self.fact_checks_df))

        self.fact_checks_df_filtered = self.fact_checks_df.sample(n=num, random_state=42)
        selected_fact_check_ids = set(self.fact_checks_df_filtered['fact_check_id'])

        self.mapping_df_filtered = self.mapping_df[self.mapping_df['fact_check_id']\
                                                   .isin(selected_fact_check_ids)]
        selected_post_ids = set(self.mapping_df_filtered['post_id'])

        self.posts_df_filtered = self.posts_df[self.posts_df['post_id'].isin(selected_post_ids)]


class Preprocessor:
    def __init__(self, fact_checks_file: str, posts_file: str, mapping_file: str):
        self.mapping_file = mapping_file
        self.fact_checks_file = fact_checks_file
        self.posts_file = posts_file

    def _load_data(self, ratio: float):
        # Load the datasets
        fact_check_post_mapping = pd.read_csv(self.mapping_file, encoding='ISO-8859-1')
        fact_checks = pd.read_csv(self.fact_checks_file, encoding='ISO-8859-1')
        posts = pd.read_csv(self.posts_file, encoding='ISO-8859-1')

        # Create Dataset
        self.dataset = FactCheckDataset(fact_checks, posts,
                                        fact_check_post_mapping, ratio)


    def prepare_data(self, ratio: float = 1.0):
        print("STARTING DATA PREPROCESSING")
        self._load_data(ratio)
        
        # Preprocess data here
        # passing empty for now

        # Returning complete dataset
        print("FINISHING DATA PREPROCESSING")
        return self.dataset


class Evaluator:
    def __init__(self, fact_checks_file: str, posts_file: str, mapping_file: str, ratio: float = 1.0):
        print("INITIALISING MODEL EVALUATOR")
        self.preprocessor = Preprocessor(fact_checks_file,
                                         posts_file, mapping_file)
        self.dataset = self.preprocessor.prepare_data(ratio)
        print("INITIALISED MODEL EVALUATOR")


    def evaluate(self, model, k : int = 3):
        """
        Evaluate the model on Success@K and Mean Reciprocal Rank (MRR).
        Args:
        - model: The fact-check retrieval model.
        - dataset (FactCheckDataset) which will be used for
            posts_df: DataFrame containing social media posts.
            mapping_df: DataFrame containing the ground truth fact-check mappings.
        - k (int): The number of top predictions to consider for Success@K and MRR.
        Returns:
        - A dictionary containing Success@K and MRR.
        """

        success_at_k_count = 0
        reciprocal_ranks = []
        posts_df = self.dataset.training_data["posts_df"]

        # Create a mapping of post_id to ground truth fact-check_ids
        post_to_ground_truth = {row['post_id']: row['fact_check_id']
                                for _, row in self.dataset.training_data["mapping_df"].iterrows()}

        # Iterate over all the posts
        for _, post_row in tqdm(posts_df.iterrows()):
            post_id = post_row['post_id']
            text = post_row['text']

            # Ensure that 'text' is a string before passing to the model
            if not isinstance(text, str):
                continue

            predictions = model.predict(text, k)
            ground_truth_fact_check_id = post_to_ground_truth.get(post_id)
            if ground_truth_fact_check_id is not None:
                predicted_fact_check_ids = [prediction['fact_check_id'] for prediction in predictions]
                if ground_truth_fact_check_id in predicted_fact_check_ids:
                    actual_rank = predicted_fact_check_ids.index(ground_truth_fact_check_id) + 1
                    reciprocal_ranks.append(1 / actual_rank)

                    # If the ground truth is within the top K, count it as success@K
                    if actual_rank <= k:
                        success_at_k_count += 1
                else:
                    # If ground truth is not found, assign a rank based on total predictions
                    reciprocal_ranks.append(0)  # No reciprocal rank if not found

        # Calculate Success@K
        success_at_k = success_at_k_count / len(posts_df)
        # Calculate Mean Reciprocal Rank (MRR)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

        return {"Success@K": success_at_k, "MRR": mrr}
