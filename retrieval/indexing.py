import torch
import os
import pickle
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils.config import BM25_INDEX_PATH, FINETUNED_MODEL_PATH

def build_bm25_index(corpus: dict):
    """Builds and saves a BM25 index from the passage corpus."""
    if os.path.exists(BM25_INDEX_PATH):
        print("BM25 index already exists. Skipping build.")
        return

    print("\nBuilding BM25 index on passages...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def bm25_tokenizer(text):
        return [word for word in text.lower().split() if word.isalnum() and word not in stop_words]

    tokenized_passages = [bm25_tokenizer(p["text"]) for p in corpus.values()]


    #it will do tf-idf and other calculations internally
    bm25 = BM25Okapi(tokenized_passages)
    
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

def encode_passages(corpus: dict, model_path: str, device: str) -> 'torch.Tensor':
    """Encodes all passages into dense vector embeddings using a specified model.
    this embedding will be used to calculate the cosine similarity between query and passage embeddings."""
    
    print(f"\nEncoding all passages with model: {model_path}...")
    model = SentenceTransformer(model_path, device=device)
    passage_texts = [p["text"] for p in corpus.values()]
    passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
    return passage_embeddings
