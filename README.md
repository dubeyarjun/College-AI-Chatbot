# College AI Chatbot for Academic Support

A beginner-friendly AI chatbot that answers college-related questions (admission, courses, exam schedule, fees, hostel, placement, contact) using Python, NLP, and Flask.

## Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This creates `model.pkl` and `vectorizer.pkl` in the project folder.

### 3. Run the chatbot

```bash
python app.py
```

### 4. Open in browser

Go to: **http://127.0.0.1:5000**

---

## Project Structure

```
college_chatbot/
├── app.py              # Flask backend, /chat API
├── train.py            # NLP training script
├── intents.json        # Dataset (intents, patterns, responses)
├── model.pkl           # Trained classifier (after train.py)
├── vectorizer.pkl      # TF-IDF vectorizer (after train.py)
├── templates/index.html
├── static/style.css
├── static/script.js
├── requirements.txt
└── README.md
```

## Tech Stack

- Python, Flask, NLTK, Scikit-learn, HTML, CSS, JavaScript, JSON

## Features

- Intent detection using TF-IDF + Logistic Regression
- Random response selection per intent
- Confidence threshold and default reply for unknown questions
- Simple web UI with responsive layout
