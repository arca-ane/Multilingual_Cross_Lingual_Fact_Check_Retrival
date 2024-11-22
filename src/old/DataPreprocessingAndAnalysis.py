import pandas as pd
import numpy as np
import re
import ast
from transformers import XLMRobertaTokenizer
import matplotlib.pyplot as plt

fact_checks_df = pd.read_csv('data/fact_checks.csv')
posts_df = pd.read_csv('data/posts.csv', encoding='ISO-8859-1')
mapping_df = pd.read_csv('data/fact_check_post_mapping.csv')

# Example preview of the data
print(fact_checks_df.head())
print(posts_df.head())
print(mapping_df.head())

# Check for missing values by columnz
print("Missing values by column:")
print(fact_checks_df.isnull().sum())
print(posts_df.isnull().sum())
print(mapping_df.isnull().sum())

# Check for missing values by row
print("\nMissing values by row (fact_checks_df):")
fact_checks_df['missing_count'] = fact_checks_df.isnull().sum(axis=1)
print(fact_checks_df[['missing_count']].value_counts())

print("\nMissing values by row (posts_df):")
posts_df['missing_count'] = posts_df.isnull().sum(axis=1)
print(posts_df[['missing_count']].value_counts())

print("\nMissing values by row (mapping_df):")
mapping_df['missing_count'] = mapping_df.isnull().sum(axis=1)
print(mapping_df[['missing_count']].value_counts())

# Fill rows with missing values in 'title' for fact_checks_df
fact_checks_df['title'].fillna(np.nan, inplace=True)

# Drop rows with missing values in 'text' for posts_df
posts_df.dropna(subset=['text'], inplace=True)

# Check for missing values by column
print("Missing values by column:")
print(fact_checks_df.isnull().sum())
print(posts_df.isnull().sum())
print(mapping_df.isnull().sum())

# Extract English translation from claim
fact_checks_df['claim_text'] = fact_checks_df['claim'].apply(lambda x: x[1] if isinstance(x, tuple) else x)

# Extract English translation from title, and use claim if title is NaN
fact_checks_df['title_text'] = fact_checks_df.apply(
    lambda row: row['title'][1] if isinstance(row['title'], tuple) and row['title'][1] is not None
                else row['claim_text'], axis=1
)

# Extract English translation from posts
posts_df['post_text'] = posts_df['text'].apply(lambda x: x[1] if isinstance(x, tuple) else x)

# Preview the results
print(fact_checks_df[['claim_text', 'title_text']].head())
print(posts_df[['post_text']].head())

# # Extract English translations from OCR and concatenate with the post text (if available)
# posts_df['ocr_text'] = posts_df['ocr'].apply(lambda x: " ".join([ocr[1] for ocr in eval(x)]) if pd.notnull(x) else "")
# posts_df['post_text'] = posts_df.apply(lambda row: row['post_text'] + " " + row['ocr_text'], axis=1)

# # Handle verdicts (optional)
# posts_df['verdicts'] = posts_df['verdicts'].apply(lambda x: eval(x) if pd.notnull(x) else [])

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the cleaning function to both the post and claim texts
fact_checks_df['claim_text'] = fact_checks_df['claim_text'].apply(clean_text)
fact_checks_df['title_text'] = fact_checks_df['title_text'].apply(clean_text)
posts_df['post_text'] = posts_df['post_text'].apply(clean_text)

# Preview cleaned text
print(fact_checks_df[['claim_text', 'title_text']].head())
print(posts_df[['post_text']].head())

# Some error with installaion of Roberta
# # Load XLM-R tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# # Tokenize and pad the sequences for both claims and posts
# fact_checks_df['claim_tokens'] = fact_checks_df['claim_text'].apply(lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids'])
# posts_df['post_tokens'] = posts_df['post_text'].apply(lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids'])

# # Example of tokenized outputs
# print(fact_checks_df['claim_tokens'].head())
# print(posts_df['post_tokens'].head())

# Calculate the numbers
num_fact_checks = fact_checks_df.shape[0]
num_posts = posts_df.shape[0]
num_mappings = mapping_df.shape[0]
#num_languages = mappings['pair_lang'].nunique()  # Number of unique language pairs

print(f"Number of Fact-Checks: {num_fact_checks}")
print(f"Number of Social Media Posts: {num_posts}")
print(f"Number of Mappings: {num_mappings}")

# Merge DataFrames without including 'pair_lang'
fact_checks_with_mapping = pd.merge(mapping_df, fact_checks_df, on='fact_check_id', how='left')
full_df = pd.merge(fact_checks_with_mapping, posts_df, on='post_id', how='left')

# Example: Count of fact-checks and posts
fact_check_count = full_df['fact_check_id'].nunique()
post_count = full_df['post_id'].nunique()
print(f"Number of fact-checks: {fact_check_count}")
print(f"Number of posts: {post_count}")

# Example: Bar chart for number of fact-checks and posts
full_df['fact_check_id'].value_counts().plot(kind='bar', figsize=(12, 6))
plt.title('Number of Posts per Fact-Check')
plt.xlabel('Fact-Check ID')
plt.ylabel('Number of Posts')
plt.show()

# Example: Merge and analyze relationships
merged_df = full_df[['fact_check_id', 'post_id']]
print(merged_df.head())
merged_df.to_csv('merged_data.csv', index=False)

# Number of posts per fact-check
posts_per_fact_check = full_df['fact_check_id'].value_counts()
print(posts_per_fact_check)

# Example: Check unique verdicts
exploded_verdicts = posts_df['verdicts'].explode()
unique_verdicts = posts_df['verdicts'].explode().unique()
print(unique_verdicts)

# Example: Preview texts
print(full_df[['claim', 'title', 'text']].head())
# Count occurrences of each unique verdict
verdict_counts = exploded_verdicts.value_counts()
print("\nVerdict Counts:")
print(verdict_counts)

# Example: Check the first few instances
print(fact_checks_df['instances'].head())
print(posts_df['instances'].head())

# Load the CSV files into DataFrames
fact_checks_df = pd.read_csv('data/fact_checks.csv')
posts_df = pd.read_csv('data/posts.csv', encoding='ISO-8859-1')
mapping_df = pd.read_csv('data/fact_check_post_mapping.csv')

# Function to extract language code from the provided format in claims, titles, and OCR/text fields
def extract_language(text):
    try:
        # Convert string representation of list to actual list/tuple
        lang_info = ast.literal_eval(text)
        if isinstance(lang_info, list) and lang_info:
            return lang_info[0][2]  # Extract the language code
    except (ValueError, IndexError, SyntaxError, TypeError):
        return None
    return None

def language(text):
    try:
        # Convert string representation of list to actual list/tuple
        lang_info = ast.literal_eval(text)
        if isinstance(lang_info, list) and lang_info:
            return lang_info[0][0]  # Extract the language code
    except (ValueError, IndexError, SyntaxError, TypeError):
        return None
    return None

# Extract language from claims in fact checks
fact_checks_df['claim_lang'] = fact_checks_df['claim'].apply(language)

# Extract language from OCR in posts
posts_df['ocr_lang'] = posts_df['ocr'].apply(extract_language)

# Merge fact checks and posts language information with the mapping DataFrame
merged_df = mapping_df.merge(fact_checks_df[['fact_check_id', 'claim_lang']], on='fact_check_id', how='left')
merged_df = merged_df.merge(posts_df[['post_id', 'ocr_lang']], on='post_id', how='left')

# Combine the languages into a pair for the pair_lang column
merged_df['pair_lang'] = merged_df.apply(lambda row: (row['claim_lang'], row['ocr_lang']), axis=1)

# Save the updated DataFrame to a new CSV file
merged_df.to_csv('updated_fact_check_post_mapping.csv', index=False)

print("Updated CSV with pair_lang column saved as 'updated_fact_check_post_mapping.csv'.")

