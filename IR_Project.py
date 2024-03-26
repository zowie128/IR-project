#%%
!pip install python-terrier
import pyterrier as pt
if not pt.started():
    pt.init()

from transformers import BertModel
import torch
import torch.nn as nn

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from pathlib import Path
import os

# Ensure NLTK data is downloaded only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
#%%
# Example of loading a dataset from PyTerrier
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')

topics = dataset.get_topics()
topics.head()

qrels = dataset.get_qrels()
qrels.head()

corpus_iter = dataset.get_corpus_iter()

# Convert to an iterator
corpus_iterator = iter(corpus_iter)

first_doc = next(corpus_iterator)
#%%
class BertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Align the layer names with the checkpoint
        self.classification = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classification(pooled_output)
        return logits

# Initialize the model
model = BertClassifier()

# Load the checkpoint
model_path = './model/doc_baseline.ckpt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the state_dict, assuming the state dict is saved under 'state_dict'
model_state_dict = checkpoint['state_dict']

# Adjust keys in the state_dict as needed
adjusted_model_state_dict = {}
for key, value in model_state_dict.items():
    # Remove the 'model.' prefix and handle the 'classifier' to 'classification' naming difference
    new_key = key.replace('model.', '').replace('classification', 'classifier')
    if 'position_ids' not in new_key:  # Ignore 'bert.embeddings.position_ids'
        adjusted_model_state_dict[new_key] = value

# Load the adjusted state dict
model.load_state_dict(adjusted_model_state_dict, strict=False)
model.eval()  # Set the model to evaluation mode

#%%
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str):
        return word_tokenize(text)

    def normalize(self, tokens):
        return [re.sub(r'\W+', '', token.lower()) for token in tokens if re.sub(r'\W+', '', token.lower())]

    def stopping(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def stemming(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, query: str, use_tokenize=True, use_normalize=True, use_stopping=True, use_stemming=False):
        if query is None or not isinstance(query, str):
            raise ValueError("Input query must be a non-empty string")
        tokens = self.tokenize(query) if use_tokenize else query.split()
        if use_normalize: tokens = self.normalize(tokens)
        if use_stopping: tokens = self.stopping(tokens)
        if use_stemming: tokens = self.stemming(tokens)
        return ' '.join(tokens)


def preprocess_queries(query: str) -> str:
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(query, use_tokenize=True, use_normalize=True, use_stopping=True, use_stemming=False)
#%%
class QueryRewriter(pt.transformer.TransformerBase):
    def __init__(self, model, tokenizer):
        super(QueryRewriter, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def transform(self, queries):
        rewritten_queries = []
        for query in queries:
            # Tokenize the query using the BERT tokenizer
            inputs = self.tokenizer.encode_plus(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Predict with the model
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output
                predictions = torch.argmax(logits, dim=-1)

            # Determine if the query should be rewritten based on model's prediction
            if predictions.item() == 1:
                rewritten_query = self.rewrite_query(query)
            else:
                rewritten_query = query

            rewritten_queries.append(rewritten_query)
        return rewritten_queries


    def rewrite_query(self, query):
        # Implement your query rewriting logic here
        rewritten_query = query
        return rewritten_query

#%% md
# ## Evaluating Rewritten vs. Original Queries with Doc MSMARCO Reranked Models
# 
# ### Objective
# 
# The goal is to utilize the four doc_msmarco reranked models to compare the performance of rewritten queries against original queries.
# 
# ### Location of Reranked Models
# 
# The reranked models are stored within the `doc_msmarco` directory, under the `model` folder. Here's the structure:
# 
# ```
# model/
# ├── doc_baseline.ckpt
# ├── doc_davinci_doc_context.ckpt
# └── wetransfer_model-1-doc2query-all-rankers_2024-03-21_1327/
#     ├── best_rank_list/
#     │   └── doc_msmarco/
#     │       ├── doc_attention_2_davinci_final_3e-4.tsv
#     │       ├── doc_davinci03_doc_context_400tok_5e-4.tsv
#     │       ├── doc_linear_2_davinci_final_1e-4.tsv
#     │       └── doc_query_2_doc_3e-4.tsv
#     └── doc_query2doc.ckpt
# ```
# 
# ### Procedure
# 
# 1. **Retrieve Documents**: Initially, documents are fetched from the MS MARCO datasets for the purpose of reranking.
# 
# 2. **Reranking**: Utilize the retrieved documents along with the queries (both original and rewritten) to perform reranking using the models specified above.
# 
# 3. **Evaluation**: Assess the reranking process to determine the effectiveness of rewritten queries in comparison to the original ones.
#%% md
# ### Step 1: Indexing the Corpus
# 
# Before we can retrieve and rerank documents based on queries, we need to ensure that the documents are indexed. You've mentioned creating an index but commented out the relevant code. Here's how you can do it properly with Pyterrier
#%% md
# 
#%%
index_location = str(Path("index").absolute())
index_exists = os.path.isfile(os.path.join(index_location, "data.properties"))

if not index_exists:
    indexer = pt.IterDictIndexer(index_location)
    index_ref = indexer.index(corpus_iter)
    print("Indexing completed.")
else:
    print("Index already exists, loading from disk.")
    index_ref = index_location
#%% md
# ### Step 2: Retrieving Documents
# 
# You need to use the created index to retrieve documents based on both the original and rewritten queries. You have already outlined this step correctly, but make sure that `index` is correctly initialized using the `index_ref` from the indexing step:
#%%
# Initialize index from the created index reference
index = pt.IndexFactory.of(index_ref)

# Retrieve documents for both original and rewritten queries using BM25
original_res = pt.BatchRetrieve(index, wmodel="BM25").transform(original_queries)
rewritten_res = pt.BatchRetrieve(index, wmodel="BM25").transform(rewritten_queries)

#%% md
# ### Step 3: Reranking the Results
# 
# After retrieving the initial set of documents for both the original and rewritten queries, the next step is to rerank these results. You've outlined the process, assuming you have a `reranker` defined. This reranker could be any model or method that takes a set of retrieved documents and reorders them based on relevance. Let's proceed with the assumption:
#%%
# TODO: implement reranking using the models
original_reranked = reranker.transform(original_res)
rewritten_reranked = reranker.transform(rewritten_res)

#%% md
# ### Step 4: Evaluating the Results
# 
# Finally, to evaluate the effectiveness of the rewritten queries versus the original queries, you'll use PyTerrier's evaluation functionalities. You need to compare the reranked results against the relevance judgments (`qrels`) using various metrics:
#%%
# Evaluate both sets of reranked results using the same relevance judgments
evaluator = pt.Evaluator()
metrics = ["map", "ndcg", "P_10"]  # Example metrics: Mean Average Precision, nDCG, Precision at 10
results = evaluator.evaluate([original_reranked, rewritten_reranked], qrels, metrics=metrics)

print(results)

#%%
