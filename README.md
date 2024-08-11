# Semi-Supervised-Text-Analysis-Using-LLMs
## Overview
This project explores the use of pre-trained Large Language Models (LLMs) to cluster text data from the 20 Newsgroups dataset. By leveraging embeddings from multiple LLMs and applying the COP-Kmeans algorithm, we aim to achieve meaningful text clustering.

## Dataset
The 20 Newsgroups dataset, with 20,000 documents across 20 categories, serves as a benchmark for text classification and clustering.

## Methodology

- **Embedding Extraction**: Used six LLMs (e.g., DeBERTa, LongFormer) to generate sentence embeddings, followed by dimensionality reduction.
- **Clustering**: Applied the COP-Kmeans algorithm with must-link and cannot-link constraints derived from cosine similarity.
- **Evaluation**: The Silhouette Score was used to evaluate clustering quality, with scores of -0.26 and 0.03 in two experiments.

## Challenges
- Limited data usage due to computational constraints.
- Issues with constraint handling in COP-Kmeans.
- Difficulty integrating LIME-XAI with LLM embeddings for text generation.


## References
- Bradley, P. S., Bennett, K. P., & Demiriz, A. (2000). "Constrained k-means clustering."
- Babaki, B., Guns, T., & Nijssen, S. (2014). "Constrained clustering using column generation.
