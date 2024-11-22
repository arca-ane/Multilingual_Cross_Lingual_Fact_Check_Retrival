import pandas as pd

class Preprocessor:
    def __init__(self, mapping_file, fact_checks_file, posts_file):
        self.mapping_file = mapping_file
        self.fact_checks_file = fact_checks_file
        self.posts_file = posts_file

    def load_data(self):
        # Load the datasets
        self.fact_check_post_mapping = pd.read_csv(self.mapping_file)
        self.fact_checks = pd.read_csv(self.fact_checks_file)
        self.posts = pd.read_csv(self.posts_file)

    def prepare_data(self):
        # Create expanded post dataset
        posts_expanded = pd.merge(self.fact_check_post_mapping[['post_id']], self.posts[['post_id', 'text']], on='post_id')
        #fact_checks_expanded = pd.merge(self.fact_check_post_mapping[['fact_check_id']], self.fact_checks[['fact_check_id', 'claim']], on='fact_check_id')
        
        # Returning the basic merged dataset for now
        return posts_expanded


preprocessor = Preprocessor("data/trial_data_mapping.csv", "data/trial_fact_checks.csv", "data/trial_posts.csv")
preprocessor.load_data()
prepared_data = preprocessor.prepare_data()
