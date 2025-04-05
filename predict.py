import tensorflow as tf
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load resources
model = tf.keras.models.load_model('models/sentiment_model.keras')
with open('models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Text preprocessing (must match training)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')])

def predict_sentiment(text):
    # Preprocess → Tokenize → Pad → Predict
    clean_text = preprocess(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=100)
    pred = model.predict(padded)
    return label_encoder.inverse_transform([np.argmax(pred)])[0], float(np.max(pred))

# Example usage
if __name__ == "__main__":
    sample_text = "The company reported excellent quarterly results"
    sentiment, confidence = predict_sentiment(sample_text)
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")