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
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing (must match train.py)
import string
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
def ensure_nltk():
    nltk_data_dir = '/tmp/nltk_data'
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    try:
        # Check if punkt tokenizer is available in any of the nltk.data.path directories
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Download punkt into the writable /tmp directory
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

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

def get_intent_by_tag(tag):
    """Return the intent dict for the given tag, or None."""
    for intent in intents_data:
        if intent.get('tag') == tag:
            return intent
    return None

def get_responses_for_tag(tag):
    """Return list of response strings for the given intent tag (legacy format with top-level responses)."""
    intent = get_intent_by_tag(tag)
    if intent is None:
        return []
    return intent.get('responses', [])

def get_response_by_best_matching_entry(tag, user_message, user_vec):
    """
    For the given intent tag, find the entry whose patterns best match the user message
    (by cosine similarity), then return a random response from that entry.
    Returns (response_text or None, confidence).
    """
    intent = get_intent_by_tag(tag)
    if intent is None:
        return None, 0.0
    entries = intent.get('entries', [])
    if not entries:
        # Legacy format: only top-level responses
        responses = intent.get('responses', [])
        if responses:
            return random.choice(responses), 1.0
        return None, 0.0
    best_score = -1.0
    best_responses = []
    for entry in entries:
        patterns = entry.get('patterns', [])
        responses = entry.get('responses', [])
        if not patterns or not responses:
            continue
        for pattern in patterns:
            cleaned_p = preprocess_text(pattern)
            if not cleaned_p:
                continue
            pattern_vec = vectorizer.transform([cleaned_p])
            sim = cosine_similarity(user_vec, pattern_vec)[0, 0]
            if sim > best_score:
                best_score = float(sim)
                best_responses = responses
    if best_responses:
        return random.choice(best_responses), best_score
    return None, 0.0

def predict_and_respond(user_message):
    """
    Predict intent, then pick the response that matches the user's question best
    (pattern-specific response). Uses confidence threshold and default when needed.
    """
    if model is None or vectorizer is None:
        return "Error: Model not loaded. Please run train.py first.", 0.0
    cleaned = preprocess_text(user_message)
    if not cleaned:
        return DEFAULT_RESPONSE, 0.0
    user_vec = vectorizer.transform([cleaned])
    probas = model.predict_proba(user_vec)[0]
    predicted_tag = model.predict(user_vec)[0]
    classes = model.classes_
    try:
        idx = list(classes).index(predicted_tag)
        confidence = float(probas[idx])
    except (ValueError, IndexError):
        confidence = 0.0
    if confidence < CONFIDENCE_THRESHOLD:
        return DEFAULT_RESPONSE, confidence
    # Get response that matches the specific question (entry with best-matching pattern)
    response_text, match_score = get_response_by_best_matching_entry(predicted_tag, user_message, user_vec)
    if response_text:
        return response_text, max(confidence, match_score)
    # Fallback to any response for this intent (legacy)
    responses = get_responses_for_tag(predicted_tag)
    if responses:
        return random.choice(responses), confidence
    return DEFAULT_RESPONSE, confidence

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

# Try to load artifacts globally so Vercel serverless environment can access them
try:
    load_artifacts()
except Exception as e:
    print(f"Failed to load artifacts at startup: {e}")

if __name__ == '__main__':
    # Run instructions (comment): Install deps (pip install -r requirements.txt),
    # Train (python train.py), Run (python app.py), Open http://127.0.0.1:5000
    app.run(debug=True, host='0.0.0.0', port=5000)
