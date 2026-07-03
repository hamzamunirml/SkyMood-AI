"""
Text Preprocessing Module
Cleans raw review/tweet text for sentiment analysis.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    """Download required NLTK resources if not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str, min_token_len: int = 3) -> str:
    """
    Clean a single piece of review/tweet text:
    - lowercase
    - remove URLs
    - remove @mentions
    - remove hashtag symbol (keep the word)
    - remove punctuation and digits
    - remove stopwords
    - lemmatize remaining tokens

    Args:
        text: raw input text
        min_token_len: drop tokens shorter than this length

    Returns:
        cleaned, space-joined string of tokens
    """
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)               # mentions
    text = re.sub(r"#", "", text)                    # hashtag symbol, keep word
    text = re.sub(r"[^a-z\s]", " ", text)             # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) >= min_token_len
    ]
    return " ".join(tokens)


def clean_series(text_series):
    """Apply clean_text() to a pandas Series of raw text."""
    return text_series.apply(clean_text)
