import os
import pickle
import random
import sys
import torch
from datetime import timedelta
import time

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from utils.data_utils import prepare_training_data_pairs
from utils.config import PROCESSED_DATA_PATH, BASE_MODEL, FINETUNED_MODEL, TRAIN_BATCH_SIZE, NUM_EPOCHS

def main():
    """
    Loads the processed data and fine-tunes the SentenceTransformer model.
    """

    #inport to include or it will ask for login and bla blV
    os.environ["WANDB_DISABLED"] = "true" # Disable Weights & Biases logging
    
    #Yuvraj --> no GPU 😭
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Processed Data ---
    """
    We won't be making index and tokenizing again if there's some error in fine tuning or we want to change some hyperparameter
    """


    print("Loading processed data...")
    with open(os.path.join(PROCESSED_DATA_PATH, "corpus_passages.pkl"), "rb") as f:
        corpus = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "queries.pkl"), "rb") as f:
        queries = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "qrels.pkl"), "rb") as f:
        qrels = pickle.load(f)



    # --- 2. Prepare Training & Validation Data ---
    """Prepares training data for fine-tuning the SentenceTransformer model using MultipleNegativesRankingLoss.
    Each query is paired with its relevant passages from the corpus based on qrels."""
    train_examples = prepare_training_data_pairs(corpus, queries, qrels)

    if not train_examples:
        print("No training examples found. Exiting.")
        return  


    # --- 3. Fine-Tune the Model ---
    """
    we'll make sure to save the best model based on validation performance. i.e save and then check if next is best or not.
    """


    print("\n--- Starting Model Fine-Tuning with Validation & Smart Batching ---")
    start_time = time.time()
    
    model = SentenceTransformer(BASE_MODEL)
    train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=TRAIN_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Create a validation set
    query_ids = list(queries.keys())
    random.shuffle(query_ids)
    val_split_idx = int(len(query_ids) * 0.9)
    val_qids = query_ids[val_split_idx:]
    val_queries = {qid: queries[qid] for qid in val_qids}
    val_qrels = {qid: qrels[qid] for qid in val_qids if qid in qrels}

    evaluator = InformationRetrievalEvaluator(val_queries, corpus, val_qrels, 
                                              name="nfcorpus-validation")
    
    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=NUM_EPOCHS,
              warmup_steps=warmup_steps,
              evaluator=evaluator,
            #   evaluation_steps=int(len(train_dataloader) * 0.1),
              output_path=FINETUNED_MODEL,
              save_best_model=True,
              show_progress_bar=True)
                   
    end_time = time.time()
    print(f"--- Fine-tuning complete in {str(timedelta(seconds=end_time - start_time))} ---")
    print(f"Best model saved to '{FINETUNED_MODEL}'")

if __name__ == "__main__":
    main()
