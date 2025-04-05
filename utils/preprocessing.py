# utils/preprocessing.py
import re

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()  # Replaces nltk.word_tokenize
    stopwords = set([
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'this', 'that', 'for', 'on', 'with', 'as', 'it', 'by', 'at', 'from'
    ])
    return ' '.join([word for word in tokens if word not in stopwords])
