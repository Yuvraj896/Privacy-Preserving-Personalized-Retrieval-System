# evaluate_bm25.py
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from bm25_retriever import BM25RetrieverFromList
import numpy as np
from tqdm import tqdm

# -----------------------------
# Load dataset
# -----------------------------
print("[INFO] Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data
labels = newsgroups.target
target_names = newsgroups.target_names

train_docs, test_docs, train_labels, test_labels = train_test_split(
    docs, labels, test_size=0.2, random_state=42
)
print(f"[INFO] Train docs: {len(train_docs)}, Test docs: {len(test_docs)}")

# -----------------------------
# Build BM25 index
# -----------------------------
print("[INFO] Building BM25 index on training set...")
retriever = BM25RetrieverFromList(train_docs)

# -----------------------------
# Evaluation metrics
# -----------------------------
def precision_at_k(y_true, y_pred, k):
    relevant = [1 if train_labels[i] == y_true else 0 for i in y_pred[:k]]
    return sum(relevant) / k

def recall_at_k(y_true, y_pred, k):
    total_relevant = sum(1 for lbl in train_labels if lbl == y_true)
    retrieved_relevant = sum(1 for i in y_pred[:k] if train_labels[i] == y_true)
    return retrieved_relevant / total_relevant if total_relevant else 0

def average_precision(y_true, y_pred):
    rels = [1 if train_labels[i] == y_true else 0 for i in y_pred]
    precisions = [sum(rels[:i+1])/(i+1) for i in range(len(rels)) if rels[i] == 1]
    return np.mean(precisions) if precisions else 0.0

def ndcg_at_k(y_true, y_pred, k):
    gains = [1 if train_labels[i] == y_true else 0 for i in y_pred[:k]]
    dcg = sum([g / np.log2(i+2) for i, g in enumerate(gains)])
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum([g / np.log2(i+2) for i, g in enumerate(ideal_gains)])
    return dcg / idcg if idcg > 0 else 0

# -----------------------------
# Evaluate BM25 retrieval
# -----------------------------
top_k = 100
precisions, recalls, maps, ndcgs = [], [], [], []

print("[INFO] Evaluating BM25 retrieval (this may take a few minutes)...")

for q_doc, q_label in tqdm(zip(test_docs[:500], test_labels[:500]), total=500):
    pred_indices, _ = retriever.search(q_doc, top_k=top_k)
    precisions.append(precision_at_k(q_label, pred_indices, top_k))
    recalls.append(recall_at_k(q_label, pred_indices, top_k))
    maps.append(average_precision(q_label, pred_indices))
    ndcgs.append(ndcg_at_k(q_label, pred_indices, top_k))

# -----------------------------
# Results
# -----------------------------
print("\n=== BM25 Evaluation on 20 Newsgroups (500 queries) ===")
print(f"Precision@{top_k}: {np.mean(precisions):.4f}")
print(f"Recall@{top_k}:    {np.mean(recalls):.4f}")
print(f"MAP:               {np.mean(maps):.4f}")
print(f"nDCG@{top_k}:      {np.mean(ndcgs):.4f}")
