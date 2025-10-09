# In models/centralized_model.py

import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
# Make sure to import PROCESSED_TEST_CSV as well
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, PROCESSED_TRAIN_CSV, PROCESSED_TEST_CSV, EMBEDDING_MODEL_NAME

def generate_document_embeddings():
    """
    Loads BOTH processed train and test data, combines them, converts text to embeddings,
    and saves the unified embeddings + doc_ids for retrieval.
    """
    # Load BOTH train and test data
    df_train = pd.read_csv(PROCESSED_TRAIN_CSV)
    df_test = pd.read_csv(PROCESSED_TEST_CSV)
    
    # Combine them into a single, unified dataframe
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Safety check for empty text
    df['text'] = df['text'].fillna("")

    # This number should now be ~18,846
    print(f"Loaded {len(df)} total documents for embedding generation.")

    # Load SentenceTransformer model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("SentenceTransformer model loaded.")

    # Generate embeddings for the combined dataframe
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    # Normalize L2
    faiss.normalize_L2(embeddings)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Save embeddings
    np.save(DOC_EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {DOC_EMBEDDINGS_PATH}")

    # Save the combined and unified doc_id mapping
    doc_ids = df['doc_id'].tolist()
    with open(DOC_IDS_PATH, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"Document IDs saved to {DOC_IDS_PATH}")

if __name__ == "__main__":
    generate_document_embeddings()