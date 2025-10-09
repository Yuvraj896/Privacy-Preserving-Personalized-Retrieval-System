import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, PROCESSED_DATA_PATH, PROCESSED_TRAIN_CSV, PROCESSED_TEST_CSV


def load_20newsgroups(subset= 'train'):

    #returns a list of dic : {'text': ..., 'label': ...}
    data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    
    documents = []
    for i in range(len(data.data)):
        text = data.data[i]
        label = data.target[i]  # numeric label for the category
        documents.append({'text': text, 'label': label})
    
    return documents


# Preprocess Text
def preprocess_text(text):
    if not isinstance(text, str):
        text = "" 

    text = text.lower()  # lowercase
    text = ' '.join(text.split())  # remove extra spaces / newlines
    return text


# for saving the dataset
def save_processed_data(documents, save_path='data/processed/processed_20news.csv'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tmp_save_path = save_path + ".tmp"
    
    data_list = []
    for idx, doc in enumerate(documents):
        processed_text = preprocess_text(doc['text'])
        data_list.append({
            'doc_id': idx,
            'text': processed_text,
            'label': doc['label']
        })
    
    df = pd.DataFrame(data_list)
    df.to_csv(tmp_save_path, index=False)
    
    # Only rename once fully written
    os.replace(tmp_save_path, save_path)
    print(f"Processed data saved to {save_path}")


#pipeline
def prepare_20newsgroups_dataset():
    
    #Loads train + test sets, combines them, and saves processed CSV
    
    train_docs = load_20newsgroups(subset='train')
    test_docs = load_20newsgroups(subset='test')
    
    all_docs = train_docs + test_docs
    for doc in all_docs:
        doc['text'] = preprocess_text(doc['text'])

    #split
    train_docs_processed = all_docs[:len(train_docs)]
    test_docs_processed  = all_docs[len(train_docs):]

    save_processed_data(train_docs_processed, PROCESSED_TRAIN_CSV)
    save_processed_data(test_docs_processed, PROCESSED_TEST_CSV )




#run
prepare_20newsgroups_dataset()