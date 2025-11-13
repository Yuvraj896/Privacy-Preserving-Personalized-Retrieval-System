import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from models.reranker import RerankerMLP
from clients.client import PairDataset  # reuse your dataset class
from config import EMBEDDER_NAME
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # federated_IR folder
MODEL_PATH = ROOT / "global_model.pth"
VAL_CSV = ROOT / "data" / "validation" / "val.csv"
CACHE_DIR = ROOT / "data" / "validation" / "cache"
BATCH_SIZE = 32
DEVICE = "cpu"  # or "cuda" if available

# ---------- LOAD MODEL ----------
model = RerankerMLP(emb_dim=384)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- PRECOMPUTE EMBEDDINGS ----------
CACHE_DIR.mkdir(exist_ok=True, parents=True)
q_cache = CACHE_DIR / "q_emb.npy"
d_cache = CACHE_DIR / "d_emb.npy"
lab_cache = CACHE_DIR / "labels.npy"

if q_cache.exists() and d_cache.exists() and lab_cache.exists():
    q_embs = np.load(q_cache)
    d_embs = np.load(d_cache)
    labels = np.load(lab_cache)
else:
    df = pd.read_csv(VAL_CSV)
    queries = df['query'].astype(str).tolist()
    docs = df['doc'].astype(str).tolist()
    labels = df['label'].astype(int).to_numpy()
    embedder = SentenceTransformer(EMBEDDER_NAME)
    q_embs = embedder.encode(queries, convert_to_numpy=True, batch_size=16)
    d_embs = embedder.encode(docs, convert_to_numpy=True, batch_size=16)
    np.save(q_cache, q_embs)
    np.save(d_cache, d_embs)
    np.save(lab_cache, labels)

# ---------- DATA LOADER ----------
val_ds = PairDataset(torch.tensor(q_embs), torch.tensor(d_embs), labels)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- EVALUATION ----------
criterion = torch.nn.MSELoss()
total_loss = 0.0
correct = 0
with torch.no_grad():
    for q_emb, d_emb, lbls in val_loader:
        q_emb = q_emb.float().to(DEVICE)
        d_emb = d_emb.float().to(DEVICE)
        lbls = lbls.to(DEVICE)
        out = model(q_emb, d_emb)
        loss = criterion(out, lbls)
        total_loss += loss.item() * q_emb.size(0)
        pred = torch.round(out)
        correct += (pred == lbls).sum().item()

mse = total_loss / len(val_loader.dataset)
accuracy = correct / len(val_loader.dataset)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
