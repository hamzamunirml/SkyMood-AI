"""
Prediction Module
Loads a saved model + vectorizer and predicts sentiment for new/custom text.
"""

import joblib
from src.preprocess import clean_text


def load_artifacts(model_path: str, vectorizer_path: str):
    """Load the trained model and TF-IDF vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer) -> str:
    """Predict the sentiment label for a single piece of raw text."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction


def predict_batch(texts, model, vectorizer):
    """Predict sentiment labels for a list of raw texts."""
    cleaned = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned)
    return model.predict(features)


if __name__ == "__main__":
    # Example usage (run from project root: python -m src.predict)
    model, vectorizer = load_artifacts("models/best_model.pkl", "models/tfidf_vectorizer.pkl")

    examples = [
        "The flight was delayed for 5 hours and staff was extremely rude.",
        "Amazing service, the crew was so friendly and helpful!",
        "The flight departed and arrived on time, nothing special.",
    ]

    for text in examples:
        sentiment = predict_sentiment(text, model, vectorizer)
        print(f"'{text}' -> {sentiment}")
