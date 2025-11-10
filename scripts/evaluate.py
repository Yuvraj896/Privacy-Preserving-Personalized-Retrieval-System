import os
import json
import pickle
import torch
from sentence_transformers import SentenceTransformer
from beir.retrieval.evaluation import EvaluateRetrieval

from utils.config import PROCESSED_DATA_PATH, BM25_INDEX_PATH, FINETUNED_MODEL, BASE_MODEL, EVAL_K_VALUES, CHUNK_NUM_SENTENCES, CHUNK_OVERLAP
from retrieval.indexing import setup_dense_retriever
from retrieval.search_bm25 import search_bm25
from retrieval.search_dense import run_dense_search
from utils.data_utils import chunk_text
from utils.evaluation import evaluate_and_plot


def main():
    """
    Loads all models and data, runs BM25, Base Dense, and Fine-Tuned Dense retrievals,
    prints evaluation metrics (nDCG, P, Recall, MAP) for all @k values,
    and generates comparison plots.
    """

    # --- 1. Device Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # --- 2. Load Data and Artifacts ---
    print("\n📦 Loading all artifacts for final evaluation ...")
    with open(os.path.join(PROCESSED_DATA_PATH, "original_corpus.pkl"), "rb") as f:
        orignal_corpus = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "passage_to_doc_id.pkl"), "rb") as f:
        passage_to_doc_id_map = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "corpus_passages.pkl"), "rb") as f:
        corpus = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "queries.pkl"), "rb") as f:
        queries = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "qrels.pkl"), "rb") as f:
        qrels = pickle.load(f)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    print(f"✅ Loaded corpus: {len(corpus)} passages, {len(queries)} queries, {len(qrels)} qrels.")

    # --- 3. Create Evaluator ---
    evaluator = EvaluateRetrieval()
    all_results = {}

    # ===================================================================
    # 🧱 1. BM25 Retrieval (Baseline)
    # ===================================================================
    print("\n🔍 Running BM25 retrieval ...")
    bm25_results = search_bm25(queries, bm25, orignal_corpus, k=max(EVAL_K_VALUES))
    all_results['BM25 (Passages)'] = bm25_results

    # ===================================================================
    # 🧠 2. Base Dense Retriever (Pretrained)
    # ===================================================================
    print(f"\n⚙️ Encoding passages with base model '{BASE_MODEL}' ...")
    base_model, base_passage_emb = setup_dense_retriever(corpus, BASE_MODEL, device)

    print("\n🔍 Running Base Dense retrieval ...")
    
    # run_dense_search expects: (queries, model, passage_embeddings, passage_to_doc_id_map)
    base_dense_results = run_dense_search(queries, base_model, base_passage_emb, passage_to_doc_id_map)
    all_results['Base Dense (Passages)'] = base_dense_results

    # ===================================================================
    # 🎯 3. Fine-Tuned Dense Retriever
    # ===================================================================
    print(f"\n🏋️ Setting up and encoding passages with fine-tuned model from '{FINETUNED_MODEL}' ...")
    # setup_dense_retriever returns (model, passage_embeddings)
    fine_tuned_model, fine_tuned_passage_emb = setup_dense_retriever(corpus, FINETUNED_MODEL, device)

    print("\n🔍 Running Fine-Tuned Dense retrieval ...")
    fine_tuned_dense_results = run_dense_search(queries, fine_tuned_model, fine_tuned_passage_emb, passage_to_doc_id_map)
    all_results['Fine-Tuned Dense (Passages)'] = fine_tuned_dense_results

    # ===================================================================
    # 📊 4. Evaluate & Plot All Models
    # ===================================================================
    evaluate_and_plot(qrels, all_results)


if __name__ == "__main__":
    main()


