# utils/evaluation.py

import os
import json
import matplotlib.pyplot as plt
import matplotlib.style as style
from beir.retrieval.evaluation import EvaluateRetrieval
from .config import PLOTS_PATH, EVAL_K_VALUES

def evaluate_and_plot(qrels: dict, results_dict: dict):
    """
    Runs evaluation for a dictionary of models and their results,
    then prints and plots the final comparison.
    """
    evaluator = EvaluateRetrieval()
    all_scores = {}
    for model_name, results in results_dict.items():
        print(f"\n--- Evaluating {model_name} ---")
        scores = evaluator.evaluate_custom(qrels, results, EVAL_K_VALUES)
        all_scores[model_name] = scores

    # --- Print Final Results ---
    print("\n" + "="*80)
    print("FINAL COMPARATIVE RESULTS")
    print("="*80)
    for model_name, scores in all_scores.items():
        print(f"\n--- {model_name} ---")
        print(f"nDCG@10: {scores['nDCG@10']:.4f}")
        print(f"Precision@10: {scores['P@10']:.4f}")
        print(f"Recall@10: {scores['Recall@10']:.4f}")

    # --- Plotting ---
    os.makedirs(PLOTS_PATH, exist_ok=True)
    style.use('seaborn-v0_8-talk')
    model_names = list(all_scores.keys())
    ndcg_10_scores = [s['nDCG@10'] for s in all_scores.values()]
    
    plt.figure(figsize=(10, 7))
    bars = plt.bar(model_names, ndcg_10_scores, color=['#4285F4', '#FBBC05'])
    plt.ylabel('nDCG@10 Score')
    plt.title('Search Model Performance Comparison (nDCG@10)')
    plt.ylim(0, max(ndcg_10_scores) * 1.25 if ndcg_10_scores else 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    chart_path = os.path.join(PLOTS_PATH, 'final_comparison.png')
    plt.savefig(chart_path)
    print(f"\nGenerated results chart: {chart_path}")

