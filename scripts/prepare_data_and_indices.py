import os
import sys
import pickle
import nltk
# # Add parent directory to path to allow imports from other folders
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
nltk.download('punkt_tab')

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

from utils.config import DATASET_NAME, PROCESSED_DATA_PATH, CHUNK_NUM_SENTENCES, CHUNK_OVERLAP
from utils.data_utils import chunk_text
from retrieval.indexing import build_bm_index


def download_and_process_data():
  # Downloads, chunks, and saves the dataset

    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = beir_util.download_and_unzip(url, "datasets")

    print(f"Loading corpus from: {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


    # need for rankbm
    corpus_docs_list = [corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "") for doc_id in corpus]
    print(f"  → Sample corpus_docs_list[0]: {corpus_docs_list[0][:150]}...\n")

    print("\nChunking documents into passages...")
    passage_corpus = []
    passage_to_doc_id_map = []
    for doc_id, doc_data in corpus.items():
        full_text = doc_data.get("title", "") + " " + doc_data.get("text", "")
        doc_chunks = chunk_text(full_text)
        for chunk in doc_chunks:
            passage_corpus.append(chunk)
            passage_to_doc_id_map.append(doc_id)

    print(f"Created {len(passage_corpus)} passages from {len(corpus)} documents.")



    # --- 3. Save All Processed Data to Disk ---
    # This is the crucial step that was missing.
    print(f"\nSaving processed data to '{PROCESSED_DATA_PATH}'...")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    with open(os.path.join(PROCESSED_DATA_PATH, "corpus_passages.pkl"), "wb") as f:
        pickle.dump(passage_corpus, f)
    with open(os.path.join(PROCESSED_DATA_PATH, "queries.pkl"), "wb") as f:
        pickle.dump(queries, f)
    with open(os.path.join(PROCESSED_DATA_PATH, "qrels.pkl"), "wb") as f:
        pickle.dump(qrels, f)

    print("Processed data saved successfully.")

    return corpus, corpus_docs_list, passage_corpus, passage_to_doc_id_map



def main():
    print("\n--- Downloading and Processing Data ---")
    corpus, corpus_docs_list, passage_corpus, passage_to_doc_id_map = download_and_process_data()
    # --- 4. Build and Save BM25 Index ---
    # This function is in your retrieval/indexing.py file
    build_bm_index(passage_corpus)
    
    print("\n--- Data preparation and BM25 indexing complete. ---")
    print("You can now run 'python training/fine_tune.py'")

if __name__ == "__main__":
    main()

