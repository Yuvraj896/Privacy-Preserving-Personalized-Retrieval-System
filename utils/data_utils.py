import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers.readers import InputExample
nltk.download('punkt_tab')

# Ensure the 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk 'punkt' tokenizer for sentence splitting...")
    nltk.download('punkt')

def chunk_text(text, num_sentences=5, overlap=2):
    sentences = sent_tokenize(text)
    if not sentences: return []
    chunks = []
    step = max(1,num_sentences - overlap)
    for i in range(0, len(sentences), step):
        chunks.append(" ".join(sentences[i:i + num_sentences]))
    return chunks


def prepare_training_data_pairs(corpus, queries, qrels):
    print("\nPreparing training data (positive pairs) for fine-tuning...")
    train_examples = []
    for query_id, query_text in queries.items():
        relevant_doc_ids = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        if not relevant_doc_ids: continue

        for doc_id in relevant_doc_ids:
            positive_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
            train_examples.append(InputExample(texts=[query_text, positive_text]))

    print(f"Created {len(train_examples)} (query, positive_passage) pairs.")
    return train_examples