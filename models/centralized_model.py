import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, PROCESSED_DATA_PATH, PROCESSED_TRAIN_CSV

def generate_document_embeddings():
    """
    Loads processed 20 Newsgroups data, converts text to embeddings,
    and saves embeddings + doc_ids for retrieval.
    """
    # Load processed CSV

    df = pd.read_csv(PROCESSED_TRAIN_CSV)

    #safety : no float value (expects string)
    df['text'] = df['text'].fillna("")

    print(f"Loaded {len(df)} documents for embedding generation.")

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

    # 3Generate embeddings
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Save embeddings
    np.save(DOC_EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {DOC_EMBEDDINGS_PATH}")

    # Save doc_id mapping
    doc_ids = df['doc_id'].tolist()
    with open(DOC_IDS_PATH, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"Document IDs saved to {DOC_IDS_PATH}")

# Optional: Run as main

if __name__ == "__main__":
    generate_document_embeddings()