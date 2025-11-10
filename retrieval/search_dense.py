import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util


def run_dense_search(queries, model, passage_embeddings, passage_to_doc_id_map):
    results = {}
    for query_id, query_text in queries.items():
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, passage_embeddings)[0]
        top_passage_results = torch.topk(cos_scores, k=min(200, len(passage_embeddings)))

        unique_doc_scores = {}
        for score, idx in zip(top_passage_results.values, top_passage_results.indices):
            doc_id = passage_to_doc_id_map[idx.item()]
            if doc_id not in unique_doc_scores:
                unique_doc_scores[doc_id] = score.item()

        results[query_id] = unique_doc_scores
    return results