import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, TOP_K
from utils.data_utils import preprocess_text 

import pandas as pd


""" Know this to understand the code
    we made dense vector embeddings for every document in train set using Sentence Transformer
    then stored those inside FAISS (fast vector search engine)

    FAISS index position → which document (ID) it represents.
"""

# index = contains all document embeddings in a compressed and searchable structure.
# doc_ids = keeps the mapping


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    
    #load
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("FAISS index loaded.")

    if not os.path.exists(DOC_IDS_PATH):
        raise FileNotFoundError(f"Document IDs file not found: {DOC_IDS_PATH}")
    
    # Load doc_ids from the pkl file
    with open(DOC_IDS_PATH, 'rb') as f:
        doc_ids = pickle.load(f)
    print(f"{len(doc_ids)} document IDs loaded.")

    return index, doc_ids


def load_embedding_model():
    #Loads a pre-trained Sentence-BERT (SBERT) model, e.g. all-MiniLM-L6-v2.
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading embedding model: {e}")


def search_query(query_text, index, doc_ids, model, doc_id_to_text, doc_id_to_label, top_k=TOP_K):
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

    if isinstance(query_text, np.ndarray):
        query_embedding = query_text
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
    else:
        processed_query = preprocess_text(query_text)
        if processed_query.lower() == "nan":
            raise ValueError("Query text is 'nan' after preprocessing!")
        
        # Encode query text to get its embedding
        query_embedding = model.encode([processed_query], convert_to_numpy=True)

        #normalize
        faiss.normalize_L2(query_embedding)  # in-place normalization

    
    # Search FAISS index
    """ this will search the query_embedding with all compressed doc vectors as FAISS idx, returns the index and distance of the most close docs"""

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id = doc_ids[idx]
        results.append({
            'doc_id': doc_id,
            'score': float(dist),
            'index': idx, # optional: position in embeddings array
            'text': doc_id_to_text.get(doc_id, "Text not found"),  # Use .get for safety
            'label': doc_id_to_label.get(doc_id, "Label not found")

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
        print(f"DocID: {r['doc_id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"index: {r['index']}")
        print(f"text: {r['text']}")
        print(f"label: {r['label']}")
        print("\n")


    query = "computer graphics image rendering"
    index , train_doc_ids = load_faiss_index()
    results = search_query(query, index, train_doc_ids, model)

    print("\nTop Results:")
    for r in results:
        print(f"DocID: {r['doc_id']}, Score: {r['score']:.4f}")
    
#     FAISS index loaded.
# 11314 document IDs loaded.

# Top Results:
# DocID: 8148, Score: 0.7120
# DocID: 9041, Score: 0.9142
# DocID: 4589, Score: 1.0087
# DocID: 8378, Score: 1.0161
# DocID: 11082, Score: 1.0183
