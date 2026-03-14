"""
College AI Chatbot - Flask Backend
==================================
Loads trained model and vectorizer, accepts user messages via POST,
predicts intent using TF-IDF + Logistic Regression, and returns a
random response from the matching intent.

RUN: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import json
import random
import pickle

from flask import Flask, request, jsonify, render_template

# Preprocessing (must match train.py)
import string
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

ensure_nltk()

app = Flask(__name__)

# Global variables for model, vectorizer, and intents (loaded once at startup)
model = None
vectorizer = None
intents_data = []

# Default response when intent is unknown or confidence is low (BONUS)
DEFAULT_RESPONSE = (
    "I'm sorry, I didn't understand that. "
    "Please ask about admissions, courses, exams, fees, hostel, placement, or contact."
)
# Confidence threshold: below this we return default response (BONUS)
# For a small educational dataset, using 0.0 ensures the best predicted
# intent is always returned instead of the default message.
CONFIDENCE_THRESHOLD = 0.0

def load_artifacts():
    """Load model.pkl, vectorizer.pkl, and intents.json from the app directory."""
    global model, vectorizer, intents_data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
    intents_path = os.path.join(base_dir, 'intents.json')
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "model.pkl or vectorizer.pkl not found. Run: python train.py"
        )
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(intents_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        intents_data = data.get('intents', [])

def preprocess_text(text):
    """Same preprocessing as in train.py: lowercase, tokenize, remove punctuation."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and t.isalnum()]
    return " ".join(tokens)

def get_responses_for_tag(tag):
    """Return list of response strings for the given intent tag."""
    for intent in intents_data:
        if intent.get('tag') == tag:
            return intent.get('responses', [])
    return []

def predict_and_respond(user_message):
    """
    Convert user message to TF-IDF vector, predict intent, and return
    a random response from that intent. Uses confidence threshold and
    default response when needed.
    """
    if model is None or vectorizer is None:
        return "Error: Model not loaded. Please run train.py first.", 0.0
    cleaned = preprocess_text(user_message)
    if not cleaned:
        return DEFAULT_RESPONSE, 0.0
    X = vectorizer.transform([cleaned])
    # Get probability estimates for each class
    probas = model.predict_proba(X)[0]
    predicted_idx = model.predict(X)[0]
    # predicted_idx is the class label; get its index for probability
    classes = model.classes_
    try:
        idx = list(classes).index(predicted_idx)
        confidence = float(probas[idx])
    except (ValueError, IndexError):
        confidence = 0.0
    if confidence < CONFIDENCE_THRESHOLD:
        return DEFAULT_RESPONSE, confidence
    responses = get_responses_for_tag(predicted_idx)
    if not responses:
        return DEFAULT_RESPONSE, confidence
    # Random response selection (BONUS)
    chosen = random.choice(responses)
    return chosen, confidence

@app.route('/')
def index():
    """Serve the main chatbot page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    API route: accepts JSON with 'message' key, returns JSON with
    'response' and optionally 'confidence'.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'response': DEFAULT_RESPONSE,
                'confidence': 0.0,
                'error': 'No JSON body'
            }), 400
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({
                'response': "Please type a message.",
                'confidence': 1.0
            }), 200
        response_text, confidence = predict_and_respond(user_message)
        return jsonify({
            'response': response_text,
            'confidence': round(confidence, 4)
        }), 200
    except Exception as e:
        return jsonify({
            'response': DEFAULT_RESPONSE,
            'confidence': 0.0,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    load_artifacts()
    # Run instructions (comment): Install deps (pip install -r requirements.txt),
    # Train (python train.py), Run (python app.py), Open http://127.0.0.1:5000
    app.run(debug=True, host='0.0.0.0', port=5000)
