import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers.readers import InputExample

# Ensure the 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk 'punkt' tokenizer for sentence splitting...")
    nltk.download('punkt')

def chunk_text(text: str, num_sentences: int, overlap: int) -> list[str]:
    """Splits a long text into smaller, overlapping chunks of sentences to fit model input limits.(2 chunked sentences can have meaning
    we capture it by having another chunk with overlap of sentences)"""

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    step = num_sentences - overlap
    if step <= 0:
        step = 1 # Ensure we always move forward

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + num_sentences])
        chunks.append(chunk)
    
    return chunks


def prepare_training_data(queries: dict, qrels: dict, corpus: dict) -> list:

    """
    Logic : Each query is paired with its relevant passages from the corpus based on qrels.
    first search the relevant doc id for the query,
    then find the passage id, if the doc id of passage is relevant we say the passage is relevant and create a pair.
    """
    doc_to_passages = {}
    for passage_id in corpus.keys():
        doc_id = '-'.join(passage_id.split('-')[:-1])
        if doc_id not in doc_to_passages:
            doc_to_passages[doc_id] = []
        doc_to_passages[doc_id].append(passage_id)

    # Step 2: Create training examples efficiently using the map
    train_examples = []
    for query_id, query_text in queries.items():
        relevant_doc_ids = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        if not relevant_doc_ids:
            continue

        for doc_id in relevant_doc_ids:
            # Fast lookup instead of a slow, nested loop
            if doc_id in doc_to_passages:
                for passage_id in doc_to_passages[doc_id]:
                    passage_text = corpus[passage_id]["text"]
                    train_examples.append(InputExample(texts=[query_text, passage_text]))

                    
    print(f"Created {len(train_examples)} (query, positive_passage) pairs.")
    return train_examples