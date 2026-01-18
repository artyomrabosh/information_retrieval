# Information retrieval

A hybrid search engine combining text and vector search with two-stage ranking (L1 + L2). 


Prerequisites
- Python 3.12+
- uv

Installation:

```bash
git clone git@github.com:artyomrabosh/information_retrieval.git
cd information_retrieval
uv sync
```

Running:
1) Download dataset from notebooks/load_dataset.ipynb or load your own
2) Precompute embeddings using notebooks/vector_search.ipynb
3) Train and save FAISS index with embeddings

```bash
uv run scipts/train_faiss_index.py
```
4) Run app.py
```bash
uv run web/app.py
```
