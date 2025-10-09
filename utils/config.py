# File paths
RAW_DATA_PATH = "data/raw/20news.csv"           # Not used now, we use sklearn fetch
PROCESSED_DATA_PATH = "data/processed/processed_20news.csv"

PROCESSED_TRAIN_CSV = "data/processed/processed_20news_train.csv"
PROCESSED_TEST_CSV  = "data/processed/processed_20news_test.csv"

DOC_EMBEDDINGS_PATH = "data/processed/doc_embeddings.npy"
DOC_IDS_PATH = "data/processed/doc_ids.pkl"
FAISS_INDEX_PATH = "data/processed/faiss_index.index"

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


TOP_K = 21  # Number of documents to retrieve for a query
