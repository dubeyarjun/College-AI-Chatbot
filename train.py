"""
College AI Chatbot - Model Training Script
==========================================
This script loads intents.json, preprocesses text using NLP,
trains a Logistic Regression classifier with TF-IDF features,
and saves the model and vectorizer as pickle files.

RUN: python train.py
(Ensure intents.json is in the same directory)
"""

import json
import pickle
import string
import os

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download NLTK data if not present (required for tokenization)
def download_nltk_data():
    """Download required NLTK corpora (e.g., punkt for tokenization)."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

def load_intents(filepath='intents.json'):
    """
    Load intents from JSON file.
    Returns list of intent dicts with 'tag', 'patterns', 'responses'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"intents file not found: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('intents', [])

def preprocess_text(text):
    """
    NLP preprocessing: lowercase, tokenize, remove punctuation.
    Returns a single string of cleaned tokens (for TF-IDF input).
    """
    if not text or not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower().strip()
    # Tokenize into words
    tokens = word_tokenize(text)
    # Remove punctuation and keep only alphanumeric tokens
    tokens = [t for t in tokens if t not in string.punctuation and t.isalnum()]
    # Join back for vectorizer (TF-IDF expects string input)
    return " ".join(tokens)

def prepare_data(intents):
    """
    Extract patterns and corresponding tags from intents.
    Returns X (list of preprocessed pattern strings), y (list of tags).
    """
    X = []
    y = []
    for intent in intents:
        tag = intent.get('tag')
        patterns = intent.get('patterns', [])
        if not tag or not patterns:
            continue
        for pattern in patterns:
            cleaned = preprocess_text(pattern)
            if cleaned:
                X.append(cleaned)
                y.append(tag)
    return X, y

def main():
    """Main training pipeline: load data, train model, save artifacts."""
    print("College AI Chatbot - Training started...")
    download_nltk_data()

    # 1. Load intents
    intents = load_intents()
    if not intents:
        raise ValueError("No intents found in intents.json")
    print(f"Loaded {len(intents)} intents.")

    # 2. Prepare features and labels
    X, y = prepare_data(intents)
    if not X or not y:
        raise ValueError("No valid patterns found in intents.")
    print(f"Total samples: {len(X)}")

    # 3. TF-IDF Vectorizer (convert text to numerical features)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=1,
        stop_words='english',
        lowercase=True
    )
    X_tfidf = vectorizer.fit_transform(X)

    # 4. Train Logistic Regression classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.15, random_state=42, stratify=y
    )
    classifier = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    classifier.fit(X_train, y_train)
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # 5. Save model and vectorizer using pickle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.pkl')
    vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print("Training completed successfully.")

if __name__ == '__main__':
    main()
