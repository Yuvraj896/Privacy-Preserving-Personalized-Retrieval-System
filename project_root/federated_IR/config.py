# federated_IR/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data preparation
NUM_CLIENTS = 20
DOCS_PER_CLIENT = 400
PAIRS_PER_CLIENT = 1000

# Embedding model
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Federated training defaults
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
LR = 1e-3
BATCH_SIZE = 16
EMB_DIM = 384

SEED = 42
