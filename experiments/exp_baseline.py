import pandas as pd
import numpy as np
from retrieval.query_retrieval import load_faiss_index, load_embedding_model, search_query
from utils.metrics import precision_at_k, recall_at_k, ndcg_at_k
from utils.config import TOP_K, PROCESSED_TEST_CSV

def evaluate_baseline():
    # Load FAISS index and doc_ids
    index, train_doc_ids = load_faiss_index()

    # Load embedding model
    model = load_embedding_model()

    # Load test data
    test_df = pd.read_csv(PROCESSED_TEST_CSV)
    # Ensure test_df has columns: 'doc_id', 'text', 'label'
    
    precisions, recalls, ndcgs = [], [], []

    # Iterate over test queries
    for i, row in test_df.iterrows():
        query_text = row['text']
        ground_truth_doc_id = [row['doc_id']]  # Only one relevant doc for simplicity

        #make its embedding
        query_embedding = model.encode([query_text])[0]


        # Retrieve top-K
        results = search_query(query_embedding, index, train_doc_ids, model, top_k=TOP_K)
        retrieved_ids = [r['doc_id'] for r in results]

        # Compute metrics
        precisions.append(precision_at_k(retrieved_ids, ground_truth_doc_id, TOP_K))
        recalls.append(recall_at_k(retrieved_ids, ground_truth_doc_id, TOP_K))
        ndcgs.append(ndcg_at_k(retrieved_ids, ground_truth_doc_id, TOP_K))

    # Compute average metrics
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_ndcg = sum(ndcgs) / len(ndcgs)

    print(f"\n--- Centralized IR Baseline Metrics ---")
    print(f"Average Precision@{TOP_K}: {avg_precision:.4f}")
    print(f"Average Recall@{TOP_K}:    {avg_recall:.4f}")
    print(f"Average NDCG@{TOP_K}:      {avg_ndcg:.4f}")

if __name__ == "__main__":
    evaluate_baseline()
