# federated_IR/models/reranker.py
import torch
import torch.nn as nn
from typing import List

class RerankerMLP(nn.Module):
    """Simple MLP reranker. Input: concat(q_emb, d_emb)"""
    def __init__(self, emb_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, q_emb, d_emb):
        x = torch.cat([q_emb, d_emb], dim=-1)
        return self.net(x).squeeze(-1)

def get_model_parameters(model: nn.Module) -> List:
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {}
    for k, arr in zip(keys, parameters):
        tensor_arr = torch.as_tensor(arr, dtype=state_dict[k].dtype)
        new_state[k] = tensor_arr.clone() if tensor_arr.requires_grad else tensor_arr

    state_dict.update(new_state)
    model.load_state_dict(state_dict)
