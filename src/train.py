"""
Model Training Module
Trains and returns multiple classification models for comparison.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_candidate_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of untrained candidate models to compare.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Linear SVM": LinearSVC(random_state=random_state, max_iter=5000),
    }


def train_model(model, X_train, y_train):
    """Fit a single model on training data."""
    model.fit(X_train, y_train)
    return model


def train_all(models: dict, X_train, y_train) -> dict:
    """Train every model in the given dictionary. Returns the same dict, fitted in place."""
    for name, model in models.items():
        train_model(model, X_train, y_train)
    return models
