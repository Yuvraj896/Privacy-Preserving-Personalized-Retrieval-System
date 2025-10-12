import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk

def run_bm25_search(queries: dict, bm25: BM25Okapi, corpus: dict) -> dict:
    """Performs search using a pre-built BM25 index."""
    print("Running BM25 search...")
    results = {}
    passage_ids = list(corpus.keys())
    
    # We need a tokenizer for the query
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    def bm25_tokenizer(text):
        return [word for word in text.lower().split() if word.isalnum() and word not in stop_words]

    for q_id, q_text in queries.items():
        tokenized_query = bm25_tokenizer(q_text)
        scores = bm25.get_scores(tokenized_query)

        #sort decending and take top 200
        top_indices = np.argsort(scores)[::-1][:200]
        results[q_id] = {passage_ids[i]: scores[i] for i in top_indices}
    
    return results

def run_dense_search(queries: dict, model: SentenceTransformer, corpus: dict, passage_embeddings: torch.Tensor) -> dict:
    """Performs search with a fine-tuned dense model."""
    print("Running Fine-Tuned Dense search...")
    results = {}
    passage_ids = list(corpus.keys())
    
    for q_id, q_text in queries.items():
        query_embedding = model.encode(q_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, passage_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(200, len(corpus)))
        results[q_id] = {passage_ids[idx.item()]: score.item() for score, idx in zip(top_results.values, top_results.indices)}
        
    return results

