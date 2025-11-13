# scripts/evaluate_all.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from models.reranker import RerankerMLP, set_model_parameters
from clients.client import PairDataset
from clients.eval_metrics import compute_ranking_metrics
from config import EMBEDDER_NAME
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DEVICE = "cpu"  # "cuda" if GPU available
BATCH_SIZE = 32
K_LIST = [1, 3, 5]  # for MRR@k and NDCG@k

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_MODEL_PATH = ROOT / "global_model.pth"
CLIENT_MODELS_DIR = ROOT / "clients" / "saved_models"
VAL_CSV = ROOT / "data" / "validation" / "val.csv"
CACHE_DIR = ROOT / "data" / "validation" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD VALIDATION DATA ----------
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

val_ds = PairDataset(torch.tensor(q_embs), torch.tensor(d_embs), labels)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- QUERY INDICES ----------
# For ranking metrics, define start/end indices for each query
# Here, assuming each query has 10 documents in order in CSV
num_docs_per_query = 10
num_queries = len(labels) // num_docs_per_query
query_indices = [(i * num_docs_per_query, (i + 1) * num_docs_per_query) for i in range(num_queries)]

# ---------- HELPER FUNCTION ----------
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for q_emb, d_emb, lbls in dataloader:
            q_emb = q_emb.float().to(device)
            d_emb = d_emb.float().to(device)
            lbls = lbls.to(device)
            out = model(q_emb, d_emb)
            loss = criterion(out, lbls)
            total_loss += loss.item() * q_emb.size(0)
            pred = torch.round(out)
            correct += (pred == lbls).sum().item()

            all_preds.append(out.cpu().numpy())
            all_labels.append(lbls.cpu().numpy())

    mse = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ranking_metrics = compute_ranking_metrics(model, dataloader, device, query_indices, k_list=K_LIST)

    return mse, accuracy, ranking_metrics

# ---------- EVALUATE GLOBAL MODEL ----------
print("\n=== Evaluating Global Model ===")
global_model = RerankerMLP(emb_dim=384)
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE))
mse, acc, rank_metrics = evaluate_model(global_model, val_loader, DEVICE)
print(f"Global Model - MSE: {mse:.4f}, Accuracy: {acc:.4f}")
for k, v in rank_metrics.items():
    print(f"{k}: {v:.4f}")

# ---------- EVALUATE CLIENT MODELS ----------
print("\n=== Evaluating Client Models ===")
for client_model_path in CLIENT_MODELS_DIR.glob("client_*.pth"):
    client_id = client_model_path.stem.split("_")[-1]
    client_model = RerankerMLP(emb_dim=384)

    # Load checkpoint safely
    checkpoint = torch.load(client_model_path, map_location=DEVICE)

    # Extract state_dict, handle DP-wrapped models
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip "_module." prefix if present
        new_key = k.replace("_module.", "") if k.startswith("_module.") else k
        new_state_dict[new_key] = v

    # Load into plain RerankerMLP
    client_model.load_state_dict(new_state_dict, strict=True)

    # Optionally print DP config if saved
    dp_conf = checkpoint.get("dp_config", None)
    if dp_conf:
        print(f"[Client {client_id}] DP config: {dp_conf}")

    # Evaluate client
    mse, acc, rank_metrics = evaluate_model(client_model, val_loader, DEVICE)
    print(f"\nClient {client_id} - MSE: {mse:.4f}, Accuracy: {acc:.4f}")
    for k, v in rank_metrics.items():
        print(f"{k}: {v:.4f}")
