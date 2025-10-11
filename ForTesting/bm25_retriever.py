# bm25_retriever.py
import re
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already available
nltk.download('stopwords', quiet=True)

class BM25RetrieverFromList:
    def __init__(self, docs):
        """
        BM25 Retriever built directly from a list of documents.
        """
        self.stop_words = set(stopwords.words('english'))
        self.docs = docs
        self.tokenized_corpus = [self._preprocess_text(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _preprocess_text(self, text):
        """
        Lowercase, remove punctuation, and remove stopwords.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if w not in self.stop_words and len(w) > 2]

    def search(self, query, top_k=10):
        """
        Search for top_k documents relevant to the query.
        Returns indices and scores.
        """
        tokenized_query = self._preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return ranked_indices, [scores[i] for i in ranked_indices]
