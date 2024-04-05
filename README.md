# Enhancing Search Query Clarity and Effectiveness through Large Language Model Query Rewriting

## Abstract

This study addresses the challenge of ambiguous or under-specified user queries in traditional information retrieval systems by utilizing large language models (LLMs) for query rewriting. We explore the effectiveness of prompt engineering and pre-processing techniques in improving the clarity and relevance of search queries. Using the TREC-DL-2020 dataset, we demonstrate that our approach enhances search result relevance, with implications for both practical applications and future research.

## Introduction

We investigate how LLMs can be leveraged to detect and rewrite ambiguous search queries, aiming to improve search accuracy and relevance. This involves exploring the impacts of prompt engineering and the application of pre-processing techniques on the rewritten queries.

### Research Questions

- **RQ1:** Effect of prompt engineering on LLM-generated query expansions.
- **RQ2:** Influence of pre-processing techniques on the accuracy and relevance of LLM-expanded queries.

### Contributions

Our study contributes insights on enhancing query specificity and relevance through LLMs, with a focus on prompt engineering and pre-processing techniques.

## Project Structure

Simplified to highlight the essential components:

```
.
├── README.md                                
├── final_submission_project.ipynb           # Main Jupyter notebook with analysis
├── expansion_results                        # Contains CSV files of expansion results
│   ├── length                               # Results categorized by query length
│   └── type                                 # Results by type of query expansion
├── index                                    # Directory for search index files
├── model                                    # Model directory with ranker configurations
├── queries_comparison.csv                   # Comparison of original and rewritten queries
├── results                                  # Search results using different algorithms
├── rewritten_queries                        # JSON files of rewritten queries
│   ├── length                               # Rewritten queries categorized by length
│   └── type                                 # Rewritten queries by type
└── rewritten_results.csv                    # Consolidated results of rewritten queries
```

To run the project open the `final_submission_project.ipynb` notebook in Jupyter and follow the instructions.