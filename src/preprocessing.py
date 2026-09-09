"""
src/preprocessing.py
======================
Reusable text and tabular preprocessing functions, used across the EDA,
modeling, and API code. Kept here (not duplicated in notebook cells) so the
same exact cleaning logic is guaranteed to run identically at training time
and at inference time in the deployed API - a common source of subtle bugs
when preprocessing is copy-pasted between notebooks and production code.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present (safe to call every
# import - it's a no-op if already downloaded)
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans a single piece of text following the brief's exact pipeline:
    lowercase -> remove HTML artifacts -> remove punctuation -> tokenize ->
    remove stopwords -> lemmatize.

    Returns the cleaned text as a single space-joined string (ready for
    TF-IDF or further tokenization).
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML artifacts (e.g. stray tags, &nbsp; entities)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize, remove stopwords, lemmatize
    tokens = word_tokenize(text)
    cleaned_tokens = [
        LEMMATIZER.lemmatize(tok) for tok in tokens
        if tok not in STOP_WORDS and tok.isalpha()
    ]

    return " ".join(cleaned_tokens)


def get_text_stats(text: str) -> dict:
    """
    Extracts simple length-based features from raw (uncleaned) text -
    character count and word count. Computed on the ORIGINAL text, not the
    cleaned version, since cleaning removes stopwords/punctuation which
    would distort the true length of what the customer actually wrote.
    """
    if not isinstance(text, str):
        return {"char_count": 0, "word_count": 0}
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
    }
