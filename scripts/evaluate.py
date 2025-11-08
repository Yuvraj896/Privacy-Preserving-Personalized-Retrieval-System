import os
import sys
import pickle
import torch

# Add parent directory to path to allow imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer

from utils.config import PROCESSED_DATA_PATH, BM25_INDEX_PATH, FINETUNED_MODEL_PATH
from retrieval.indexing import encode_passages
from retrieval.search import run_bm25_search, run_dense_search
from utils.evaluation import evaluate_and_plot

def main():
    """
    Loads all models and data, runs searches, and evaluates the results.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- 1. Load All Artifacts ---
    print("\n--- Loading all artifacts for final evaluation ---")
    with open(os.path.join(PROCESSED_DATA_PATH, "corpus_passages.pkl"), "rb") as f:
        corpus = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "queries.pkl"), "rb") as f:
        queries = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "qrels.pkl"), "rb") as f:
        qrels = pickle.load(f)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
        
    print(f"Loading fine-tuned model from '{FINETUNED_MODEL_PATH}'...")
    fine_tuned_model = SentenceTransformer(FINETUNED_MODEL_PATH)

    # --- 2. Encode Passages with Fine-Tuned Model ---
    passage_embeddings = encode_passages(corpus, FINETUNED_MODEL_PATH, device)

    # --- 3. Run Searches using our search module ---
    bm25_results = run_bm25_search(queries, bm25, corpus)
    dense_results = run_dense_search(queries, fine_tuned_model, corpus, passage_embeddings)

    # --- 4. Evaluate and Plot using our evaluation module ---
    final_results = {
        'BM25 (Passages)': bm25_results,
        'Fine-Tuned Dense': dense_results
    }
    evaluate_and_plot(qrels, final_results)

if __name__ == "__main__":
    main()

