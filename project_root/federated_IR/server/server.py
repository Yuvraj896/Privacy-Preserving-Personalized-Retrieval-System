# federated_IR/server/server.py
import argparse
from pathlib import Path
import sys
import torch

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reranker import RerankerMLP, set_model_parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, LR
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Global variable to store latest parameters
latest_global_params = None

def fit_metrics_agg(metrics_list):
    """
    Weighted aggregation of fit metrics from Flower clients.
    Supports metrics returned as tuples (num_examples, metrics_dict) or plain dicts.
    """
    if not metrics_list:
        return {}

    agg_metrics = {}
    total_examples = 0

    for item in metrics_list:
        # Case: tuple/list with (num_examples, metrics_dict)
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
            num_examples, metrics = item
        # Case: plain dict
        elif isinstance(item, dict):
            num_examples = 1
            metrics = item
        else:
            continue

        total_examples += num_examples
        for k, v in metrics.items():
            agg_metrics.setdefault(k, 0.0)
            agg_metrics[k] += v * num_examples  # weighted sum

    # Divide by total examples to get average
    if total_examples > 0:
        for k in agg_metrics:
            agg_metrics[k] /= total_examples

    print(f"[Server] Aggregated metrics: {agg_metrics}")
    return agg_metrics

# Subclass FedAvg to capture global parameters after aggregation
class FedAvgWithGlobalCapture(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        global latest_global_params
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            # Flower may return just Parameters OR (Parameters, metrics)
            if isinstance(aggregated, tuple):
                params, _ = aggregated
            else:
                params = aggregated
            latest_global_params = params
        return aggregated


def save_global_model(strategy, save_path="global_model.pth"):
    """
    Saves the global PyTorch model using latest captured parameters.
    """
    global latest_global_params
    if latest_global_params is None:
        print("[Server] Warning: No global parameters available yet.")
        return

    try:
        nd_params = parameters_to_ndarrays(latest_global_params)
    except Exception as e:
        print(f"[Server] Failed to convert latest_parameters: {e}")
        return

    # Convert to torch tensors
    torch_params = [torch.tensor(p, dtype=torch.float32) for p in nd_params]

    # Initialize model and set parameters
    model = RerankerMLP(emb_dim=384)
    set_model_parameters(model, torch_params)

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"[Server] Global model saved to {save_path}")

def start_server_app(server_address: str, num_rounds: int, local_epochs: int, lr: float, num_clients: int):
    """
    Start Flower server with FedAvg strategy that captures global parameters.
    """
    strategy = FedAvgWithGlobalCapture(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"epochs": local_epochs, "lr": lr},
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )

    print(f"[Server] Starting Flower server at {server_address} for {num_rounds} rounds (clients: {num_clients})")

    # Start the server (blocking call)
    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )

    # After training finishes, save the global model
    save_global_model(strategy, save_path="global_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    args = parser.parse_args()

    start_server_app(
        server_address=args.server_address,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        num_clients=args.num_clients,
    )
