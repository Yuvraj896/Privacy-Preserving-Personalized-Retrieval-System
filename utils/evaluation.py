import os
import json
import matplotlib.pyplot as plt
import matplotlib.style as style
from beir.retrieval.evaluation import EvaluateRetrieval
from utils.config import PLOTS_PATH, EVAL_K_VALUES


def evaluate_and_plot(qrels: dict, results_dict: dict):
    """
    Runs evaluation for all retrievers, prints all major metrics (nDCG, Precision, Recall, MAP)
    for each k in EVAL_K_VALUES, and plots comparative nDCG@10 results.
    """

    evaluator = EvaluateRetrieval()
    all_scores = {}

    # ===================================================================
    # 🧠 1. Evaluate Each Model
    # ===================================================================
    for model_name, results in results_dict.items():
        print(f"\n🔍 Evaluating {model_name} ...")
        scores = evaluator.evaluate(qrels, results, EVAL_K_VALUES)
        all_scores[model_name] = scores

        # Print detailed results
        print(f"\n📊 {model_name} Detailed Results:")
        print("=" * 60)
        for metric_name, metric_scores in scores.items():
            print(f"\n{metric_name.upper()} SCORES:")
            for k, value in metric_scores.items():
                print(f"  {k}: {value:.4f}")

    # ===================================================================
    # 🧾 2. Final Summary Comparison
    # ===================================================================
    print("\n" + "=" * 80)
    print("🏁 FINAL COMPARATIVE SUMMARY (nDCG, P, Recall, MAP)")
    print("=" * 80)

    for model_name, scores in all_scores.items():
        print(f"\n--- {model_name} ---")
        for k in EVAL_K_VALUES:
            ndcg = scores["nDCG"].get(f"nDCG@{k}", 0)
            precision = scores["P"].get(f"P@{k}", 0)
            recall = scores["Recall"].get(f"Recall@{k}", 0)
            map_score = scores["MAP"].get(f"MAP@{k}", 0)
            print(f"@{k} → nDCG: {ndcg:.4f} | P: {precision:.4f} | Recall: {recall:.4f} | MAP: {map_score:.4f}")

    # ===================================================================
    # 📈 3. Plotting nDCG@10 for Visual Comparison
    # ===================================================================
    os.makedirs(PLOTS_PATH, exist_ok=True)
    style.use('seaborn-v0_8-talk')

    model_names = list(all_scores.keys())
    ndcg_10_scores = [s["nDCG"].get("nDCG@10", 0.0) for s in all_scores.values()]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(model_names, ndcg_10_scores, color=['#4285F4', '#34A853', '#FBBC05'])
    plt.ylabel('nDCG@10 Score')
    plt.title('Search Model Performance Comparison (nDCG@10)')
    plt.ylim(0, max(ndcg_10_scores) * 1.25 if ndcg_10_scores else 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=12)

    chart_path = os.path.join(PLOTS_PATH, 'final_comparison.png')
    plt.savefig(chart_path)
    print(f"\n📊 Generated performance chart saved at: {chart_path}")

    print("\n✅ Evaluation complete. Metrics printed above and chart saved.\n")
