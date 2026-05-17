"""
Text Preprocessor for Mental Stress NLP Pipeline

Handles:
- Lowercasing
- URL / HTML / mention removal
- Punctuation and special character removal
- Tokenization (NLTK)
- Stopword removal (with preserved negation words)
- Lemmatization (WordNetLemmatizer)
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Download required NLTK resources (idempotent)
# ---------------------------------------------------------------------------
def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Core preprocessor
# ---------------------------------------------------------------------------

# Negation words to keep (important for sentiment)
NEGATION_WORDS = {
    "no", "not", "nor", "never", "neither", "nobody", "nothing",
    "nowhere", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
    "hasn't", "hadn't", "wouldn't", "couldn't", "shouldn't",
}

# Build stopword set minus negation words
_BASE_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = _BASE_STOPWORDS - NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove Twitter @mentions and #hashtags (but keep the word after #)
    5. Remove punctuation except apostrophes (for contractions)
    6. Collapse whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep text after # (hashtag word)
    text = re.sub(r"#(\w+)", r" \1 ", text)

    # Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am",
    }
    for short, full in contractions.items():
        text = text.replace(short, full)

    # Remove punctuation (keep spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove digits (optional — can be kept for financial stress)
    text = re.sub(r"\d+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Word tokenization using NLTK punkt."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords while preserving negation words."""
    return [t for t in tokens if t not in CUSTOM_STOPWORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using WordNet lemmatizer."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, return_tokens: bool = False):
    """
    Full preprocessing pipeline.

    Args:
        text: Raw input text
        return_tokens: If True, return list of tokens; else return joined string

    Returns:
        Preprocessed string or list of tokens
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    # Filter short tokens (< 2 chars)
    tokens = [t for t in tokens if len(t) >= 2]

    if return_tokens:
        return tokens
    return " ".join(tokens)


def preprocess_batch(texts: list[str], return_tokens: bool = False) -> list:
    """
    Preprocess a batch of texts.

    Args:
        texts: List of raw text strings
        return_tokens: If True, return list of token lists

    Returns:
        List of preprocessed strings or token lists
    """
    return [preprocess(t, return_tokens=return_tokens) for t in texts]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "I can't stop worrying about my job deadline. My heart is racing!!!",
        "Check out https://example.com — I feel so hopeless today #depression @friend",
        "I've lost all motivation. Nothing matters anymore. I don't see the point.",
    ]
    print("=== Preprocessor Test ===")
    for s in samples:
        result = preprocess(s)
        print(f"\nRaw:    {s}")
        print(f"Clean:  {result}")
