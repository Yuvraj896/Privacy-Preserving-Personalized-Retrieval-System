import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, TOP_K

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    
    #load
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("FAISS index loaded.")

    if not os.path.exists(DOC_IDS_PATH):
        raise FileNotFoundError(f"Document IDs file not found: {DOC_IDS_PATH}")
    
    # Load doc_ids
    with open(DOC_IDS_PATH, 'rb') as f:
        doc_ids = pickle.load(f)
    print(f"{len(doc_ids)} document IDs loaded.")

    return index, doc_ids


def load_embedding_model():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading embedding model: {e}")


def search_query(query_text, index, doc_ids, model, top_k=TOP_K):
    """
    Input:
        query_text : str -> user query
        index      : FAISS index
        doc_ids    : list of doc_ids
        model      : SentenceTransformer model
        top_k     : number of results
    Output:
        List of dictionaries: [{'doc_id':..., 'score':..., 'text':..., 'label':...}, ...]
    """
    # Encode query
    if not query_text.strip():
        raise ValueError("Query text is empty!")

    query_embedding = model.encode([query_text])
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id = doc_ids[idx]
        results.append({
            'doc_id': doc_id,
            'score': float(dist),
            'index': idx  # optional: position in embeddings array
        })
    return results


# example
if __name__ == "__main__":
    index, doc_ids = load_faiss_index()
    model = load_embedding_model()

    query = "Space shuttle program updates"
    results = search_query(query, index, doc_ids, model)

    print("\nTop Results:")
    for r in results:
        print(f"DocID: {r['doc_id']}, Score: {r['score']:.4f}")