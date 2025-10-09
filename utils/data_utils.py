import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os
from utils.config import DOC_EMBEDDINGS_PATH, DOC_IDS_PATH, PROCESSED_DATA_PATH, PROCESSED_TRAIN_CSV, PROCESSED_TEST_CSV
import re
from nltk.corpus import stopwords


def load_20newsgroups(subset= 'train'):

    #returns a list of dic : {'text': ..., 'label': ...}
    data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    
    documents = []
    for i in range(len(data.data)):
        text = data.data[i]
        label = data.target[i]  # numeric label for the category
        documents.append({'text': text, 'label': label})
    
    return documents


"""
Keep punctuation for structure.
Keep stopwords (since SBERT handles them).
Only lowercase + strip noise.
"""

def preprocess_text(text):

    stop_words = set(stopwords.words('english'))

    # Remove headers/footers/quotes
    text = re.sub(r'(writes:|Subject:|From:|Lines:|Article:|Organization:).*', '', text)
    text = re.sub(r'(>+).*', '', text)  # remove quoted lines

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove stopwords
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)


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
# In data_utils.py

def prepare_20newsgroups_dataset():
    """
    Loads train + test sets, combines them, assigns globally unique IDs,
    processes text, and then saves them to separate CSVs.
    """
    train_docs_raw = load_20newsgroups(subset='train')
    test_docs_raw = load_20newsgroups(subset='test')
    
    # Keep track of the split point
    num_train_docs = len(train_docs_raw)
    
    all_docs_raw = train_docs_raw + test_docs_raw
    
    data_list = []
    # Create one master list with globally unique IDs
    for idx, doc in enumerate(all_docs_raw):
        processed_text = preprocess_text(doc['text'])
        data_list.append({
            'doc_id': idx,  # This ID is now unique across the entire dataset
            'text': processed_text,
            'label': doc['label']
        })
        
    # Create a single DataFrame
    all_df = pd.DataFrame(data_list)
    
    # Split the DataFrame back into train and test sets
    train_df = all_df.iloc[:num_train_docs]
    test_df = all_df.iloc[num_train_docs:]
    
    # --- Save the data using pandas, which is simpler ---
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_CSV), exist_ok=True)
    
    train_df.to_csv(PROCESSED_TRAIN_CSV, index=False)
    print(f"Processed train data saved to {PROCESSED_TRAIN_CSV}")
    
    test_df.to_csv(PROCESSED_TEST_CSV, index=False)
    print(f"Processed test data saved to {PROCESSED_TEST_CSV}")



#run
prepare_20newsgroups_dataset()