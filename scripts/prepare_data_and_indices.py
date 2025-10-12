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
from retrieval.indexing import build_bm25_index

def main():
    """
    Downloads, chunks, and saves the dataset. Then builds and saves the BM25 index.
    """
    # --- 1. Download and Load Raw Data ---
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    data_path = beir_util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # --- 2. Chunk the Corpus ---
    """ Logic : corpus have id and data, we will join the title and text of each document,
        then chunk it into smaller passages using chunk_text function from data_utils.py.
        and we got to remember the document id for which the chunck belongs to  hence , map required.
    """

    print("\nChunking documents into passages...")
    passage_corpus = {}
    for doc_id, doc_data in corpus.items():
        full_text = doc_data.get("title", "") + " " + doc_data.get("text", "")
        doc_chunks = chunk_text(full_text, CHUNK_NUM_SENTENCES, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(doc_chunks):
            passage_id = f"{doc_id}-p{i}"
            # We store the text for retrieval and the original doc_id for mapping
            passage_corpus[passage_id] = {"text": chunk, "doc_id": doc_id}
            
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


    # --- 4. Build and Save BM25 Index ---
    # This function is in your retrieval/indexing.py file
    build_bm25_index(passage_corpus)
    
    print("\n--- Data preparation and BM25 indexing complete. ---")
    print("You can now run 'python training/fine_tune.py'")

if __name__ == "__main__":
    main()

