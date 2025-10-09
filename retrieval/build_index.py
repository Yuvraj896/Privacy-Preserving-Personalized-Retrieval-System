import faiss
import numpy as np
import pickle
import os
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, FAISS_INDEX_PATH


def build_faiss_index():
    """
    Build a FAISS index from document embeddings and save it to disk.
    """
    try :
        if not os.path.exists(DOC_EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Embeddings file not found: {DOC_EMBEDDINGS_PATH}")

        #load emb
        embeddings = np.load(DOC_EMBEDDINGS_PATH)
        print(f"Loaded embeddings with shape: {embeddings.shape}")

        # Load document IDs for creating mapping Index -> position of doc_id
        if not os.path.exists(DOC_IDS_PATH):
            raise FileNotFoundError(f"Document IDs file not found: {DOC_IDS_PATH}")
        
        
        with open(DOC_IDS_PATH, 'rb') as f:
            doc_ids = pickle.load(f)
        print(f"Loaded {len(doc_ids)} document IDs.")

        # Create FAISS index
        
        dim = embeddings.shape[1]  # embedding dimension

        index = faiss.IndexFlatIP(dim)  # exact search using L2 distance
        print(f"FAISS index created with dimension {dim}.")

        # Add embeddings to the index
        index.add(embeddings)
        print(f"Added {index.ntotal} vectors to the FAISS index.")

        # Save the index to disk
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    
    except Exception as e:
        print(f"Error while building FAISS index: {e}")

if __name__ == "__main__":
    build_faiss_index()