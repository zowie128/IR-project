import requests
import pyterrier as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from pathlib import Path
import os
import pandas as pd
import re
import json


# Ensure NLTK data is downloaded only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define file paths
original_queries_path = 'original_queries.json'
rewritten_queries_path = 'rewritten_queries.json'

# Functions to save and load data
def save_data_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def load_data_from_file(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return None

def preprocess_rewritten_queries(rewritten_queries):
    """
    Preprocesses the rewritten queries to extract useful information.

    :param rewritten_queries: A list of dictionaries with 'generated_text' keys.
    :return: A list of cleaned, concise queries.
    """
    cleaned_queries = []

    for query in rewritten_queries:
        # Extract the generated text
        generated_text = query.get('generated_text', '')

        # Remove instructional text and formatting tags
        useful_text = re.sub(r"<s> \[INST\].*?\[/INST\]</s>", "",
                             generated_text, flags=re.DOTALL)

        # Split on line breaks or common dividers and select the first non-empty line
        potential_queries = re.split(r"\n\nOR\n\n|;", useful_text)
        potential_queries = [q.strip() for q in potential_queries if q.strip()]

        # Choose the first non-empty, concise piece of text
        if potential_queries:
            cleaned_queries.append(potential_queries[0])

    return cleaned_queries

class DatasetLoader:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        if not pt.started():
            pt.init()
        self.dataset = pt.get_dataset(dataset_id)
        self.topics = self.dataset.get_topics()
        self.qrels = self.dataset.get_qrels()
        self.corpus_iter = self.dataset.get_corpus_iter()
        self.corpus_iterator = iter(self.corpus_iter)

    def get_first_doc(self):
        return next(self.corpus_iterator)

    def get_original_queries(self):
        return [topic['query'] for topic_id, topic in self.topics.iterrows()]

class QueryEvaluator:
    def __init__(self, tokenizer_model, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def evaluate_queries(self, sentences):
        features = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        print(features)
        with torch.no_grad():
            scores = self.model(**features).logits
        return scores[:, 0]

# TODO: We need to adjust this class so that it gives reasonable outputs for the queries
class RewriteQueries:
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    def __init__(self, auth_token):
        self.headers = {"Authorization": f"Bearer {auth_token}"}

    def query(self, query):
        prompt = f"<s> [INST] Concisely rewrite this into a search engine query: '{query}'. Aim for brevity and clarity without further explanation. [/INST]</s>"
        prompt = {"inputs": prompt}
        response = requests.post(self.API_URL, headers=self.headers, json=prompt)
        return response.json()

# Example usage
if __name__ == "__main__":
    # Load or fetch original queries
    original_queries = load_data_from_file(original_queries_path)
    if original_queries is None:
        dataset_loader = DatasetLoader('irds:msmarco-passage/trec-dl-2020')
        original_queries = dataset_loader.get_original_queries()
        save_data_to_file(original_queries, original_queries_path)

    print(f"Total number of queries: {len(original_queries)}")

    # Evaluate queries to determine which need rewriting
    query_evaluator = QueryEvaluator("Ashishkr/query_wellformedness_score",
                                     "Ashishkr/query_wellformedness_score")
    well_formed_scores = query_evaluator.evaluate_queries(original_queries)
    selected_threshold = 0.4
    queries_to_rewrite = [query for query, score in
                          zip(original_queries, well_formed_scores) if
                          score < selected_threshold]
    print(f"We have {len(queries_to_rewrite)} queries to rewrite.")

    # Check if rewritten queries already exist
    rewritten_queries = load_data_from_file(rewritten_queries_path)
    if rewritten_queries is None:
        rewritten_queries = []
        query_rewriter = RewriteQueries("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")
        for query in queries_to_rewrite:
            response = query_rewriter.query(query)
            if response and isinstance(response, list) and 'generated_text' in \
                    response[0]:
                generated_text = response[0]['generated_text']
                rewritten_queries.append({'generated_text': generated_text})
        save_data_to_file(rewritten_queries, rewritten_queries_path)

    # Process the rewritten queries
    cleaned_queries = preprocess_rewritten_queries(rewritten_queries)
    for query in cleaned_queries:
        print(f"Cleaned Query: {query}")







