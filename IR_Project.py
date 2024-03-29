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
import spacy
from spacy.matcher import Matcher
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token


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

def remove_redundancy(preprocessed_query):
    tokens = preprocessed_query.split()
    seen = set()
    unique_tokens = [t for t in tokens if not (t in seen or seen.add(t))]
    return ' '.join(unique_tokens)



# Load spaCy English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Extend stop words list with common but low-information words
extended_stop_words = {"define", "meaning", "example", "describe", "use", "refer", "relate", "involve", "include", "give", "take", "make", "see", "want", "get", "say", "ask", "tell", "be", "know", "do", "have", "would", "should", "could", "about"}
for word in extended_stop_words:
    STOP_WORDS.add(word)

# Customize token extension to flag important tokens to keep
Token.set_extension("is_important", default=False, force=True)

def preprocess_query(query):
    """
    Preprocess a single query using spaCy for tokenization, lemmatization, and stop word removal,
    aiming for greater conciseness.
    """
    # Process the text
    doc = nlp(query)

    # Identify important tokens to preserve
    for ent in doc.ents:
        for token in ent:
            token._.is_important = True

    for token in doc:
        if token.pos_ in {"PROPN", "NOUN", "VERB"}:
            token._.is_important = True

    # Condense the query by keeping important tokens and removing less important ones
    tokens = [token.lemma_.lower() for token in doc if (token._.is_important or token.text.lower() in extended_stop_words) and not token.is_stop and token.pos_ != "PUNCT"]

    # Reconstruct the query
    preprocessed_query = " ".join(tokens)

    return preprocessed_query





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
    dataset_loader = DatasetLoader('irds:msmarco-passage/trec-dl-2020')
    # Load or fetch original queries
    original_queries = load_data_from_file(original_queries_path)
    if original_queries is None:
        original_queries = dataset_loader.get_original_queries()
        save_data_to_file(original_queries, original_queries_path)

    # put original queries into a DataFrame
    original_queries_df = pd.DataFrame(original_queries)
    print(original_queries_df)



    print(f"Total number of queries: {len(original_queries)}")

    # Evaluate queries to determine which need rewriting
    query_evaluator = QueryEvaluator("Ashishkr/query_wellformedness_score", "Ashishkr/query_wellformedness_score")
    well_formed_scores = query_evaluator.evaluate_queries(original_queries)
    selected_threshold = 0.4
    queries_to_rewrite = [(query, score) for query, score in zip(original_queries, well_formed_scores) if score < selected_threshold]

    print(f"We have {len(queries_to_rewrite)} queries to rewrite.")

    # Modified section for handling rewritten queries to maintain original query association
    rewritten_queries = load_data_from_file(rewritten_queries_path)
    if rewritten_queries is None:
        rewritten_queries = []
        query_rewriter = RewriteQueries("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")
        for original_query, _ in queries_to_rewrite:
            response = query_rewriter.query(original_query)
            if response and isinstance(response, list) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text']
                rewritten_queries.append({'original_query': original_query, 'generated_text': generated_text})
        save_data_to_file(rewritten_queries, rewritten_queries_path)

    # Process the rewritten queries with the original query included
    cleaned_queries = [(query['original_query'], preprocess_rewritten_queries([query])[0]) for query in rewritten_queries]

    with open('cleaned_queries_with_original.json', 'w') as f:
        json.dump(cleaned_queries, f)

    for original_query, cleaned_query in cleaned_queries:
        preprocessed_query = preprocess_query(cleaned_query)
        final_query = remove_redundancy(preprocessed_query)
        # print(f"Original Query: {original_query} | Final Query: {final_query}")


    # output the final queries to a JSON file
    with open('final_queries.json', 'w') as f:
        json.dump(cleaned_queries, f)


    # Load the final queries with original and rewritten versions
    with open('final_queries.json', 'r') as f:
        queries_data = json.load(f)

    # Load the cleaned and potentially rewritten queries
    with open('cleaned_queries_with_original.json', 'r') as f:
        cleaned_queries = json.load(f)

    # Create a new DataFrame for rewritten queries, ensuring all queries are included
    rewritten_queries_list = []
    for q in original_queries:
        # Find the rewritten version if it exists, otherwise use the original
        rewritten_or_original = next((item for item in cleaned_queries if
                                      item['original_query'] == q['query']),
                                     None)
        if rewritten_or_original:
            rewritten_query = rewritten_or_original['final_query']
        else:
            rewritten_query = q[
                'query']  # Use original if not found in rewritten
        rewritten_queries_list.append(
            {'qid': q['qid'], 'query': rewritten_query})

    rewritten_queries_df = pd.DataFrame(rewritten_queries_list)


    print(rewritten_queries_df.head())

    index_location = str(Path("index").absolute())
    index_exists = os.path.isfile(
        os.path.join(index_location, "data.properties"))

    # Fetch corpus iterator just before indexing
    if not index_exists:
        corpus_iter = dataset_loader.get_corpus_iter()  # Ensure this line is correctly placed
        indexer = pt.IterDictIndexer(index_location)
        index_ref = indexer.index(corpus_iter)
        print("Indexing completed.")
    else:
        print("Index already exists, loading from disk.")
        index_ref = index_location

    # Assuming qrels are loaded correctly
    qrels = dataset_loader.qrels

    index = pt.IndexFactory.of(index_ref)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    eval_metrics = [pt.measures.RR(rel=1), pt.measures.nDCG @ 10,
                    pt.measures.MAP(rel=1)]


    #
    # # Evaluating Original Queries
    # print("Evaluating Original Queries with BM25:")
    # results_original = pt.Experiment(
    #     [bm25],
    #     original_queries_df,
    #     qrels,
    #     eval_metrics,
    #     names=["BM25 Original"]
    # )
    #
    # # Evaluating Rewritten Queries
    # print("\nEvaluating Rewritten Queries with BM25:")
    # results_rewritten = pt.Experiment(
    #     [bm25],
    #     rewritten_queries_df,
    #     qrels,
    #     eval_metrics,
    #     names=["BM25 Rewritten"]
    # )




