# federated_IR/clients/eval_metrics.py
import torch
import numpy as np

def compute_ranking_metrics(model, dataloader, device, query_indices, k_list=[1,3,5]):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for q_emb, d_emb, labels in dataloader:
            q_emb = q_emb.to(device).float()
            d_emb = d_emb.to(device).float()
            outputs = model(q_emb, d_emb)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mrr_scores = {f"MRR@{k}": [] for k in k_list}
    ndcg_scores = {f"NDCG@{k}": [] for k in k_list}

    for start, end in query_indices:
        preds = all_preds[start:end]
        labels = all_labels[start:end]

        ranking = np.argsort(-preds)
        labels_sorted = labels[ranking]

        for k in k_list:
            topk_labels = labels_sorted[:k]
            # MRR@k
            ranks = np.where(topk_labels == 1)[0]
            mrr_scores[f"MRR@{k}"].append(1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0)
            # NDCG@k
            dcg = np.sum((2 ** topk_labels - 1) / np.log2(np.arange(2, k + 2)))
            idcg = np.sum((2 ** np.sort(labels)[-k:] - 1) / np.log2(np.arange(2, k + 2)))
            ndcg_scores[f"NDCG@{k}"].append(dcg / (idcg + 1e-8))

    metrics = {}
    for k in k_list:
        metrics[f"MRR@{k}"] = np.mean(mrr_scores[f"MRR@{k}"])
        metrics[f"NDCG@{k}"] = np.mean(ndcg_scores[f"NDCG@{k}"])

    return metrics
