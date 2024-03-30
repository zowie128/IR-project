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
    # Clean up the query by removing unwanted characters
    query = re.sub(r'\n+', ' ', query)  # Replace one or more newlines with a single space
    query = re.sub(r'\s+', ' ', query).strip()  # Replace multiple spaces with a single space and trim

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


def clean_text(text):
    """Performs common cleaning operations on text."""
    text = re.sub(r"[\'\(\)]", '', text)  # Remove specific characters
    text = re.sub(r'\n+', ' ', text)  # Newlines to space
    return re.sub(r'\s+', ' ', text).strip()  # Multiple spaces to single space

def preprocess_query_final(query, max_tokens=10):
    """Preprocesses a single query by tokenizing, normalizing, removing stop words, and limiting to a maximum number of tokens."""
    if not query:
        raise ValueError("Input query must be a non-empty string")
    query = clean_text(query)
    tokens = word_tokenize(query)
    tokens = [re.sub(r'\W+', '', token.lower()) for token in tokens if re.sub(r'\W+', '', token.lower())]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = tokens[:max_tokens]
    return " ".join(tokens)


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

        # Clean up the generated text by removing unwanted characters
        generated_text = re.sub(r'\n+', ' ', generated_text)  # Replace one or more newlines with a single space
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()  # Replace multiple spaces with a single space and trim

        # Remove instructional text and formatting tags
        useful_text = re.sub(r"<s> \[INST\].*?\[/INST\]</s>", "", generated_text, flags=re.DOTALL)

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

    # Assuming each topic includes a 'query_id' and 'query' field
    def get_original_queries(self):
        return [(topic['qid'], topic['query']) for topic_id, topic in
                self.topics.iterrows()]


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
    original_queries = dataset_loader.get_original_queries()
    original_queries_df = pd.DataFrame(original_queries, columns=['qid', 'query'])

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

    # Convert the loaded data into a DataFrame
    # This assumes cleaned_queries now includes q_id in each tuple
    df_queries = pd.DataFrame(cleaned_queries,
                              columns=['original', 'rewritten'])

    # Perform a left merge to include all original queries
    df_queries_aligned = pd.merge(original_queries_df, df_queries,
                                  left_on='query', right_on='original',
                                  how='left')

    # Fill NaN values in 'rewritten' with the 'query' values
    df_queries_aligned['rewritten'].fillna(df_queries_aligned['query'],
                                           inplace=True)
    df_queries_aligned.drop('original', axis=1, inplace=True)
    df_queries_aligned.rename(columns={'query': 'original'}, inplace=True)
    df_queries_aligned.rename(columns={'rewritten': 'query'}, inplace=True)
    # Rename 'q_id' column to 'qid' to align with PyTerrier's expectations
    df_queries_aligned.rename(columns={'q_id': 'qid'}, inplace=True)

    # run preprocess final queries on df_queries_aligned: query column
    df_queries_aligned['query'] = df_queries_aligned['query'].apply(preprocess_query_final)

    print(df_queries_aligned.head())



    index_location = str(Path("index").absolute())
    index_exists = os.path.isfile(
        os.path.join(index_location, "data.properties"))

    # Fetch corpus iterator just before indexing
    if not index_exists:
        corpus_iter = dataset_loader.corpus_iter  # Adjusted from get_corpus_iter() to corpus_iter
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
    tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")

    eval_metrics = [pt.measures.RR(rel=1), pt.measures.nDCG @ 10,
                    pt.measures.MAP(rel=1)]


    # Evaluating Original Queries
    print("Evaluating Original Queries with BM25 and TF-IDF:")
    results_original = pt.Experiment(
        [bm25, tf_idf],  # List of retrieval systems to evaluate
        original_queries_df[['qid', 'query']],  # DataFrame with queries
        qrels,  # Qrels for relevance judgments
        eval_metrics,  # Evaluation metrics
        names=["BM25 Original", "TF-IDF Original"]  # Names for the systems
    )

    print(f"Results for Original Queries:\n{results_original}")
    results_original.to_csv('results_original.csv', index=False)

    print("Evaluating Rewritten Queries with BM25 and TF-IDF:")
    simple_results = pt.Experiment(
        [bm25, tf_idf],
        df_queries_aligned[['qid', 'query']],
        qrels,
        eval_metrics,
        names=["BM25 Rewritten", "TF-IDF Rewritten"]
    )

    print(f"Results for Rewritten Queries:\n{simple_results}")
    simple_results.to_csv('simple_results.csv', index=False)




