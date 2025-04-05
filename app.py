from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from utils.preprocessing import preprocess_text

app = Flask(__name__)

# Load model and utilities
model = load_model('models/sentiment_model.keras')

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('models/label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input text
        text = request.form['message']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        
        return redirect(url_for('result', sentiment=predicted_label[0]))
    
    return render_template('predict.html')

@app.route('/result')
def result():
    sentiment = request.args.get('sentiment', 'Neutral')
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)