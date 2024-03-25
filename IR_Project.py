#%%
!pip install python-terrier
import nltk
import pandas as pd

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import random
from pathlib import Path
import pyterrier as pt
if not pt.started():
    pt.init()
#%%
# Example of loading a dataset from PyTerrier
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
#%%
topics = dataset.get_topics()
topics.head()
#%%
qrels = dataset.get_qrels()
qrels.head()
#%%
corpus_iter = dataset.get_corpus_iter()

# Convert to an iterator
corpus_iterator = iter(corpus_iter)

first_doc = next(corpus_iterator)
print(first_doc)
#%% md
# The paper you've shared proposes an innovative approach for context-aware query rewriting, specifically designed for conversational search systems. This methodology addresses the common challenge of query ambiguity by dynamically understanding and incorporating the context of multi-turn conversations into the search process. Here's a breakdown of the methodology and its implementation into an algorithm:
# 
# ### Preliminaries and Background
# 
# - The primary goal of search systems is to retrieve the most relevant document for a given query from a document repository. This becomes complex in conversational search due to the multi-turn nature of queries, where the intent behind the current query is dependent on the history of previous queries.
# 
# ### Methodology Overview
# 
# 1. **CRDR Model Proposal**: The paper introduces the CRDR model, which consists of a Query Rewrite module and a Dense Retrieval module. This model aims to overcome query ambiguity by enhancing the query embedding with relevant terms identified during the query rewriting process.
# 
# 2. **Query Rewriting via Modification**: Instead of creating a new query or simply expanding the existing query with relevant terms, the CRDR model modifies the current query by replacing or inserting tokens based on the context. This process involves:
#    - Encoding the query context to understand its meaning.
#    - Tagging each token in the context with labels indicating their relevance or potential as insertion points for modification.
#    - Modifying the current query by either replacing tokens with relevant ones or inserting relevant terms at appropriate positions.
# 
# 3. **Dense Retrieval Implementation**: The model employs a dense retrieval approach where both the query and document are encoded into dense representations using a deep neural model (like BERT). The relevance between a query and document is then determined by computing the similarity between their dense embeddings.
# 
# ### Implementation Steps for an Algorithm
# 
# 1. **Encode-Tag-Modify Framework for Query Rewriting**:
#    - **Encode**: Use a pretrained language model to encode the multi-turn query context into contextualized token representations.
#    - **Tag**: Apply a token-level classification (using an MLP with Softmax) to assign each token a label indicating its relevance or role in query modification.
#    - **Modify**: Based on the tagging, modify the current query by replacing or inserting relevant terms to generate a self-contained query that accurately represents the user's intent.
# 
# 2. **Dense Retrieval with Contextualized Query Embedding**:
#    - Encode the query and documents into dense embeddings.
#    - Enhance the query embedding by integrating embeddings of relevant terms identified during the query rewriting phase. This is aimed at making the query representation more comprehensive and context-aware.
#    - Use similarity scoring between the enhanced query embedding and document embeddings to retrieve the most relevant document.
# 
# 3. **Optimization and Enhancement**:
#    - Leverage a teacher-student framework for further refining the query encoder, where a teacher model encodes a manually curated "oracle" query and a student model learns from this to encode the contextualized query more effectively.
#    - Enhance the query embedding by dynamically adjusting the influence of relevant term embeddings based on their attention scores, ensuring that all important information from the query context is captured.
# 
# ### Algorithmic Considerations
# 
# - Implementing this methodology requires a deep understanding of natural language processing (NLP) and familiarity with pretrained language models like BERT.
# - It involves sophisticated data preprocessing to manage multi-turn conversations, token classification to understand query context, and the application of neural networks for generating dense embeddings.
# - Efficient similarity scoring mechanisms (like approximate nearest neighbor search) are crucial for the retrieval phase to ensure scalability and responsiveness of the search system.
# 
# By following these steps, developers can create a context-aware query rewriting algorithm that significantly improves the performance of conversational search systems, making them more intuitive and responsive to user needs.
#%%
from pyterrier.transformer import TransformerBase
import torch
from transformers import BertTokenizer, BertModel
import faiss

class CRDRModel:
    def __init__(self, model_name='bert-base-uncased', index_path=None):
        # Initialize tokenizer and model for BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # Initialize FAISS index for dense retrieval
        # Assuming the embeddings have already been indexed in FAISS
        self.faiss_index = faiss.read_index(index_path) if index_path else None

        # Placeholder for other initializations (e.g., query history, embeddings index)

    def encode(self, texts):
        """Encodes a list of texts into contextualized embeddings."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # Use pooled output for simplicity

    def tag_and_modify(self, query, query_history):
        """Tags tokens in the query and modifies the query based on the history.
        This is a simplified placeholder. Actual implementation would involve detailed logic."""
        # Simplified logic: just return the query as is
        # In practice, use encoded context and query to determine modifications
        return query

    def retrieve_documents(self, query_embedding):
        """Performs dense retrieval given a query embedding.
        Returns document IDs and their corresponding similarity scores."""
        # Example FAISS search (top 10 results)
        D, I = self.faiss_index.search(query_embedding.cpu().detach().numpy(), 10)
        return I[0], D[0]  # IDs and distances

    def process_query(self, query, query_history):
        """Processes a single query within the context of its history."""
        modified_query = self.tag_and_modify(query, query_history)
        query_embedding = self.encode([modified_query])

        doc_ids, scores = self.retrieve_documents(query_embedding)

        # Placeholder for how to return or process the results
        return doc_ids, scores

# Assuming you have an index set up with PyTerrier for the MSMARCO dataset
index = pt.IndexFactory.of(str(Path("index").absolute()))

# Initialize the CRDR model with the index
crdr_model = CRDRModel(index_path=index)

# Example usage with a PyTerrier DataFrame
queries = pt.new.queries(["What is PyTerrier?", "Explain deep learning in IR"])
results = crdr_model(queries)

print(results)
#%%
# Instantiate the query rewriting transformer
query_rewriter = QueryRewriter()
#%%
# Transform the topics with the query rewriter
rewritten_topics = query_rewriter.transform(topics)
#%%
# Compare the original and rewritten queries
for i in range(10):
    # where i is random number
    i = random.randint(0, len(topics) - 1)
    print(f"Original: {topics['query'][i]}")
    print(f"Rewritten: {rewritten_topics['query'][i]}\n")
#%%
# indexer = pt.IterDictIndexer(str(Path("index").absolute()))
# index_ref = indexer.index(corpus_iter)
#%%
index = pt.IndexFactory.of(str(Path("index").absolute()))
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
#%%
from pyterrier.measures import RR, nDCG, MAP

# results_dir = Path("results")
# results_dir.mkdir(exist_ok=True)

pt.Experiment(
    [tf_idf, bm25],
    dataset.get_topics(),
    dataset.get_qrels(),
    names=["TF-IDF", "BM25"],
    eval_metrics=[RR @ 10, nDCG @ 20, MAP, nDCG @ 10],
)
#%%
pt.Experiment(
    [tf_idf, bm25],
    rewritten_topics,  
    dataset.get_qrels(),
    names=["TF-IDF", "BM25"],
    eval_metrics=[RR @ 10, nDCG @ 20, MAP, nDCG @ 10],
)
#%%
