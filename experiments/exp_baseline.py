import pandas as pd
import numpy as np
import os
from retrieval.query_retrieval import load_faiss_index, load_embedding_model, search_query
from utils.metrics import precision_at_k, recall_at_k, ndcg_at_k
from utils.config import TOP_K, PROCESSED_TEST_CSV
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
    # Load FAISS index and doc_ids
    index, train_doc_ids = load_faiss_index()

    # Load embedding model
    model = load_embedding_model()

    # Load test data
    test_df = pd.read_csv(PROCESSED_TEST_CSV)
    
    precisions, recalls, ndcgs = [], [], []
    debug_results = []

    for i, row in test_df.iterrows():
        # if i > 10:
        #     break

        query_text = str(row['text'])
        if query_text.lower() == "nan" or not query_text.strip():
            continue

        relevant_doc_ids = test_df[test_df['label'] == row['label']]['doc_id'].tolist()
        results = search_query(query_text, index, train_doc_ids, model, top_k=TOP_K)
        retrieved_ids = [r['doc_id'] for r in results]

        precisions.append(precision_at_k(retrieved_ids, relevant_doc_ids, TOP_K))
        recalls.append(recall_at_k(retrieved_ids, relevant_doc_ids, TOP_K))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_doc_ids, TOP_K))

        debug_results.append({
            'query_text': query_text,
            'query_doc_id': row['doc_id'],
            'query_label': row['label'],
            'retrieved_doc_ids': [r['doc_id'] for r in results],
            'retrieved_texts': [r['text'] for r in results],
            'retrieved_scores': [r['score'] for r in results],
            'relevant_doc_ids': relevant_doc_ids
        })

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_ndcg = sum(ndcgs) / len(ndcgs)

    print(f"\n--- Centralized IR Baseline Metrics ---")
    print(f"Average Precision@{TOP_K}: {avg_precision:.4f}")
    print(f"Average Recall@{TOP_K}:    {avg_recall:.4f}")
    print(f"Average NDCG@{TOP_K}:      {avg_ndcg:.4f}")

    # # Save to CSV (existing)
    # debug_file_csv = "results/debug_retrievals.csv"
    # os.makedirs(os.path.dirname(debug_file_csv), exist_ok=True)
    # pd.DataFrame(debug_results).to_csv(debug_file_csv, index=False)
    # print(f"Debugging info saved to {debug_file_csv}")

    # # Save to JSON (readable)
    # debug_file_json = "results/debug_retrievals_readable.json"
    # with open(debug_file_json, "w", encoding="utf-8") as f:
    #     json.dump(debug_results, f, indent=4, ensure_ascii=False)
    # print(f"Readable debugging info saved to {debug_file_json}")

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
