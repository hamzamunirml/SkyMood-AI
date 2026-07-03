"""
Feature Extraction Module
Converts cleaned text into numerical TF-IDF features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000, ngram_range=(1, 2)) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with sensible defaults for short review/tweet text.

    Args:
        max_features: maximum vocabulary size
        ngram_range: n-gram range, e.g. (1, 2) for unigrams + bigrams

    Returns:
        An unfitted TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )


def fit_transform_train(vectorizer: TfidfVectorizer, train_texts):
    """Fit the vectorizer on training text and transform it."""
    return vectorizer.fit_transform(train_texts)


def transform_texts(vectorizer: TfidfVectorizer, texts):
    """Transform new text using an already-fitted vectorizer."""
    return vectorizer.transform(texts)
