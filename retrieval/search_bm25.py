import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk

def remove_stopwords_from_text(text, stop_words):
    """Cleans and tokenizes text for BM25."""
    return [word for word in text.lower().split() if word.isalnum() and word not in stop_words]

def search_bm25(queries, bm25, corpus, k):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))

    results = {}
    corpus_ids = list(corpus.keys())
    for query_id, query_text in queries.items():
        tokenized_query = remove_stopwords_from_text(query_text,stop_words)
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:k+100]
        results[query_id] = {corpus_ids[idx]: float(doc_scores[idx]) for idx in top_indices}
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

