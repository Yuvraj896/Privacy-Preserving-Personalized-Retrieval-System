# federated_IR/clients/client.py
import argparse
import pandas as pd
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import json
from opacus import PrivacyEngine
# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from clients.eval_metrics import compute_ranking_metrics
from models.reranker import RerankerMLP, get_model_parameters, set_model_parameters
from config import EMBEDDER_NAME
CLIENT_MODELS_DIR = Path(__file__).parent / "saved_models"
CLIENT_MODELS_DIR.mkdir(exist_ok=True)

# Optional Opacus
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except Exception:
    OPACUS_AVAILABLE = False

def parameters_to_model(parameters, model):
    state_dict = model.state_dict()
    for (key, _), tensor in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(tensor)
    return state_dict

def model_to_parameters(model):
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]

class PairDataset(Dataset):
    def __init__(self, q_embs, d_embs, labels):
        self.q_embs = q_embs
        self.d_embs = d_embs
        self.labels = labels

    def __len__(self):
        return int(self.q_embs.shape[0])

    def __getitem__(self, idx):
        return self.q_embs[idx], self.d_embs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def precompute_embeddings(csv_path: str, embedder_name: str, cache_dir: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    q_cache = cache_dir / "q_emb.npy"
    d_cache = cache_dir / "d_emb.npy"
    lab_cache = cache_dir / "labels.npy"
    query_idx_cache = cache_dir / "query_indices.npy"

    if q_cache.exists() and d_cache.exists() and lab_cache.exists() and query_idx_cache.exists():
        q_embs = np.load(q_cache, allow_pickle=True)
        d_embs = np.load(d_cache, allow_pickle=True)
        labels = np.load(lab_cache, allow_pickle=True)
        query_indices = np.load(query_idx_cache, allow_pickle=True)
        return q_embs, d_embs, labels, query_indices

    df = pd.read_csv(csv_path)
    queries = df['query'].astype(str).tolist()
    docs = df['doc'].astype(str).tolist()
    labels = df['label'].astype(int).to_numpy()

    embedder = SentenceTransformer(embedder_name)
    q_embs = embedder.encode(queries, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    d_embs = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True, batch_size=32)

    # Group indices by query for exact MRR/NDCG
    query_indices = []
    current_q = queries[0]
    start_idx = 0
    for i, q in enumerate(queries):
        if q != current_q:
            query_indices.append((start_idx, i))
            start_idx = i
            current_q = q
    query_indices.append((start_idx, len(queries)))

    np.save(q_cache, q_embs, allow_pickle=True)
    np.save(d_cache, d_embs, allow_pickle=True)
    np.save(lab_cache, labels, allow_pickle=True)
    np.save(query_idx_cache, query_indices, allow_pickle=True)

    return q_embs, d_embs, labels, query_indices


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for q_emb, d_emb, labels in dataloader:
        q_emb = q_emb.to(device).float()
        d_emb = d_emb.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(q_emb, d_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * q_emb.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_local(model, val_loader, device):
    model.eval()
    preds = []
    labels_all = []
    with torch.no_grad():
        for q_emb, d_emb, labels in val_loader:
            q_emb = q_emb.to(device).float()
            d_emb = d_emb.to(device).float()
            outputs = model(q_emb, d_emb)
            preds.extend(outputs.cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())
    mse = np.mean((np.array(preds) - np.array(labels_all)) ** 2)
    return mse


def test(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for q_emb, d_emb, labels in dataloader:
            q_emb = q_emb.to(device).float()
            d_emb = d_emb.to(device).float()
            labels = labels.to(device)
            
            outputs = model(q_emb, d_emb)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * q_emb.size(0)
            
            pred = torch.round(outputs)
            correct += (pred == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, dp_config, query_indices, client_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.dp_config = dp_config
        self.query_indices = query_indices
        self.client_id = client_id

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Load global parameters
        self.model.load_state_dict(parameters_to_model(parameters, self.model))
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("lr", 1e-3))
        
        # ---------- Differential Privacy Setup ----------
        if self.dp_config and OPACUS_AVAILABLE:
            
            
            # Ensure model is in training mode
            self.model.train()
            
            privacy_engine = PrivacyEngine()
            self.model, optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.dp_config.get("noise_multiplier", 1.0),
                max_grad_norm=self.dp_config.get("max_grad_norm", 1.0),
            )
        else:
            privacy_engine = None

        # ---------- Training Loop ----------
        self.model.train()
        criterion = torch.nn.MSELoss()

        for epoch in range(config.get("epochs", 1)):
            running_loss = 0.0
            for q_emb, d_emb, labels in self.train_loader:
                q_emb = q_emb.float().to(self.device)
                d_emb = d_emb.float().to(self.device)
                labels = labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(q_emb, d_emb)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * q_emb.size(0)
            
            avg_loss = running_loss / len(self.train_loader.dataset)
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{config.get('epochs',1)} train loss: {avg_loss:.4f}")

        # ---------- Compute DP epsilon safely ----------
        epsilon = None
        if privacy_engine:
            try:
                # Attempt to compute DP epsilon
                epsilon = privacy_engine.get_epsilon(delta=1e-5)
                print(f"[Client {self.client_id}] ε = {epsilon:.2f}, δ = 1e-5")
            except Exception:
                # Catch all exceptions (memory-heavy or unsupported)
                print(f"[Client {self.client_id}] Skipping epsilon computation (unsupported or memory-heavy).")
        else:
            print(f"[Client {self.client_id}] Non-DP training (standard mode).")

        # Switch model to eval for metrics
        self.model.eval()
        save_dir = Path("clients/saved_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"client_{self.client_id}.pth"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "dp_config": self.dp_config
        }, save_path)
        print(f"[Client {self.client_id}] Model saved to {save_path}")

        # Return updated parameters and metrics
        metrics = {}
        if epsilon is not None:
            metrics["epsilon"] = epsilon

        return model_to_parameters(self.model), len(self.train_loader.dataset), metrics


    def evaluate(self, parameters, config):
        # Load received global parameters
        self.model.load_state_dict(parameters_to_model(parameters, self.model))
        self.model.eval()
        
        try:
            # Compute ranking metrics
            metrics = compute_ranking_metrics(
                self.model,
                self.val_loader,
                self.device,
                query_indices=self.query_indices,
                k_list=[1,3,5]
            )

            # Flower expects (loss, num_examples, metrics_dict)
            val_loss = evaluate_local(self.model, self.val_loader, self.device)
            return float(val_loss), len(self.val_loader.dataset), metrics
        except Exception as e:
            print(f"[Client {self.client_id}] Evaluation error: {e}")
            return 0.0, len(self.val_loader.dataset), {}

def start_client(server_address, data_dir, client_id, device, dp_config, embedder_name):
    # Prepare client-specific data
    client_dir = Path(data_dir) / f"client_{client_id}"
    csv_path = client_dir / "pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run data prep first.")

    # Precompute embeddings for training
    cached_dir = client_dir / "cache"
    q_embs, d_embs, labels, query_indices = precompute_embeddings(str(csv_path), embedder_name, cached_dir)

    train_ds = PairDataset(torch.tensor(q_embs), torch.tensor(d_embs), labels)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Precompute embeddings for validation (shared across clients)
    val_csv = Path(data_dir) / "validation" / "val.csv"
    val_cache_dir = Path(data_dir) / "validation" / "cache"
    qv, dv, lv, val_query_indices = precompute_embeddings(str(val_csv), embedder_name, val_cache_dir)
    val_ds = PairDataset(torch.tensor(qv), torch.tensor(dv), lv)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Initialize model
    model = RerankerMLP(emb_dim=q_embs.shape[1])

    # Create Flower client (no privacy_engine passed here)
    fl_client = FlowerClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        dp_config=dp_config,
        query_indices=val_query_indices,
        client_id=client_id
    )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=fl_client,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_noise", type=float, default=1.0)
    parser.add_argument("--dp_max_grad", type=float, default=1.0)
    parser.add_argument("--dp_sample_rate", type=float, default=0.02)
    args = parser.parse_args()

    dp_conf = None
    if args.use_dp:
        dp_conf = {
            "noise_multiplier": args.dp_noise,
            "max_grad_norm": args.dp_max_grad,
            "sample_rate": args.dp_sample_rate
        }

    start_client(args.server, args.data_dir, args.client_id, args.device, dp_conf, EMBEDDER_NAME)
