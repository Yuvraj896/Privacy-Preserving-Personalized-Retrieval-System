# --- Data & Model Settings ---
DATASET_NAME = "nfcorpus"
BASE_MODEL = 'all-mpnet-base-v2'
FINETUNED_MODEL = 'models/checkpoints/nfcorpus-finetuned'

# --- Paths ---
PROCESSED_DATA_PATH = 'data/processed'
PLOTS_PATH = 'results/plots'
BM25_INDEX_PATH = 'models/checkpoints/bm25_index.pkl'

# --- Chunking Settings ---
CHUNK_NUM_SENTENCES = 5
CHUNK_OVERLAP = 2

# --- Fine-Tuning Settings ---
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

EVAL_K_VALUES = [10,20]