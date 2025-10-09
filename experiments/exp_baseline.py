import pandas as pd
import numpy as np
import os
from retrieval.query_retrieval import load_faiss_index, load_embedding_model, search_query
from utils.metrics import precision_at_k, recall_at_k, ndcg_at_k
from utils.config import TOP_K, PROCESSED_TEST_CSV, PROCESSED_TRAIN_CSV
import json

"""
We’ll redefine relevance as:

A retrieved document is relevant if it has the same category label (e.g., “comp.graphics”, “sci.space”) as the query.

Before : we were considering A document is relevant if it has the same category (label) as the query
    ground_truth_doc_id = [row['doc_id']]
After : we will consider doc with same related doc
    relevant_doc_ids = test_df[test_df['label'] == row['label']]['doc_id'].tolist()

"""


def evaluate_baseline():
    index, all_doc_ids = load_faiss_index()
    model = load_embedding_model()
    
    precisions, recalls, ndcgs = [], [], []
    debug_results = []

    # --- Create the complete mappings here ---
    train_df = pd.read_csv(PROCESSED_TRAIN_CSV)
    test_df = pd.read_csv(PROCESSED_TEST_CSV)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    doc_id_to_text = dict(zip(all_df['doc_id'], all_df['text']))
    doc_id_to_label = dict(zip(all_df['doc_id'], all_df['label']))
    # --- Mappings are now complete ---


    # Limit to first 20 queries for fast debugging
    for i, row in test_df.iterrows():
        # if i >= 20:
        #     break

        query_text = str(row['text'])
        query_doc_id = row['doc_id'] # Get the ID of the query document
        query_label = row['label']

        if query_text.lower() == "nan" or not query_text.strip():
            continue

        relevant_doc_ids = test_df[test_df['label'] == query_label]['doc_id'].tolist()
        
        # We must also remove the query doc from the list of relevant items, 
        # as a document cannot be relevant to itself in this context.
        if query_doc_id in relevant_doc_ids:
            relevant_doc_ids.remove(query_doc_id)
        

        results = search_query(query_text, index, all_doc_ids, model, doc_id_to_text, doc_id_to_label, top_k=TOP_K)
        retrieved_ids = [r['doc_id'] for r in results if r['doc_id'] != query_doc_id]  # Exclude the query doc itself
        retrieved_ids = retrieved_ids[:TOP_K]  # Limit to TOP_K results


        precisions.append(precision_at_k(retrieved_ids, relevant_doc_ids, TOP_K))
        recalls.append(recall_at_k(retrieved_ids, relevant_doc_ids, TOP_K))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_doc_ids, TOP_K))

        # Print sample results for inspection

        if i < 5:
            print(f"\nQuery {i}: {query_text[:60]}...")
            print(f"Retrieved doc IDs: {retrieved_ids}")
            print(f"Relevant doc IDs: {relevant_doc_ids[:10]}...")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_ndcg = sum(ndcgs) / len(ndcgs)

    print(f"\n--- Centralized IR Baseline Metrics (first 20 queries) ---")
    print(f"Average Precision@{TOP_K}: {avg_precision:.4f}")
    print(f"Average Recall@{TOP_K}:    {avg_recall:.4f}")
    print(f"Average NDCG@{TOP_K}:      {avg_ndcg:.4f}")

    
if __name__ == "__main__":
    evaluate_baseline()


    # # print it to see how it actually looks
    # print("\n--- test_df Info ---")
    # test_df = pd.read_csv(PROCESSED_TEST_CSV)
    # print(test_df.head(5))
    # print(test_df.columns)
    # print(f"Total test samples: {len(test_df)}")


    """
    doc_id → unique ID
    text → cleaned article text
    label → numeric category (0 – 19 for 20 newsgroups)
    """
