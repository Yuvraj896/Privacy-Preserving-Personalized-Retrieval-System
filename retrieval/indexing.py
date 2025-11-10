import torch
import os
import pickle
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils.config import BM25_INDEX_PATH, FINETUNED_MODEL_PATH

def remove_stopwords_from_text(text,stop_words):
    """Cleans and tokenizes text for BM25."""
    return [word for word in text.lower().split() if word.isalnum() and word not in stop_words]

def build_bm_index(corpus_docs_list):
    
    # make sure to have stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    print(f"  → Total documents: {len(corpus_docs_list)}")
    
    #make tokenized corpus and save
    tokenized_corpus = [remove_stopwords_from_text(doc, stop_words) for doc in corpus_docs_list]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- Save the BM25 index ---
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"✅ BM25 index built and saved to '{BM25_INDEX_PATH}'\n")


def setup_dense_retriever(passage_corpus, model , device):
    print("\nSetting up BASE dense retriever and encoding PASSAGES...")
    print(f"Using device: {device}")
    model = SentenceTransformer(model, device=device)
    passage_embeddings = model.encode(passage_corpus, convert_to_tensor=True, show_progress_bar=True, batch_size=256)
    return model, passage_embeddings
