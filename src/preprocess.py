"""
NLP Preprocessing Pipeline for Customer Feedback
Handles text cleaning, contraction expansion, negation-aware stopword filtering,
and morphological normalization (Lemmatization / Stemming).
Completely offline-ready and robust.
"""

import re
import string

# Standard English stopwords (NLTK/Scikit-Learn baseline)
STANDARD_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}

# Negation words to explicitly PRESERVE in stopword removal
NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "barely", "hardly", "scarcely",
    "rarely", "seldom", "nothing", "nowhere", "none"
}

# Active Stopwords: Standard Stopwords minus Negation Words
FILTERED_STOPWORDS = STANDARD_STOPWORDS - NEGATION_WORDS

# Contractions mapping
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "aint": "is not",
    "arent": "are not",
    "couldnt": "could not",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hasnt": "has not",
    "havent": "have not",
    "isnt": "is not",
    "shant": "shall not",
    "shouldnt": "should not",
    "wasnt": "was not",
    "werent": "were not",
    "wont": "will not",
    "wouldnt": "would not"
}

# Irregular verb and participle normalization mapping
IRREGULAR_LEMMAS = {
    "broken": "break",
    "broke": "break",
    "breaks": "break",
    "breaking": "break",
    "frozen": "freeze",
    "froze": "freeze",
    "freezing": "freeze",
    "stolen": "steal",
    "stole": "steal",
    "lost": "lose",
    "worse": "bad",
    "worst": "bad",
    "best": "good",
    "damaged": "damag",
    "delayed": "delay",
    "crashed": "crash"
}

_NORMALIZER = None

def setup_nltk_resources():
    """Offline safe initialization. Never blocks."""
    pass

def get_normalizer():
    """
    Uses the same offline stemmer in training and inference on every machine.
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer('english')
        _NORMALIZER = stemmer.stem
    return _NORMALIZER

def expand_contractions(text: str) -> str:
    """Expands colloquial contractions (e.g. didn't -> did not)."""
    text = text.lower().replace("’", "'")
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        text = text.replace(contraction, expansion)
    return text

def clean_text(text: str, remove_stopwords: bool = True, normalize: bool = True) -> str:
    """
    Cleans raw customer feedback text:
    1. Lowers case and expands contractions
    2. Strips URLs, emails, HTML tags
    3. Keeps alphabetic characters and spaces
    4. Filters out stopwords (while strictly preserving negations like 'not')
    5. Applies Lemmatization / Stemming
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    text = expand_contractions(text)
    
    # Strip HTML & URLs
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Strip non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    tokens = text.strip().split()
    normalizer = get_normalizer() if normalize else None
    
    cleaned_tokens = []
    for token in tokens:
        token = token.lower()
        if len(token) < 2:
            continue
        if remove_stopwords and token in FILTERED_STOPWORDS:
            continue
        if normalize and token in IRREGULAR_LEMMAS:
            token = IRREGULAR_LEMMAS[token]
        elif normalize and normalizer:
            token = normalizer(token)
        cleaned_tokens.append(token)
        
    return " ".join(cleaned_tokens)

def preprocess_corpus(texts, remove_stopwords: bool = True, normalize: bool = True):
    """Batch processes an iterable of texts."""
    return [clean_text(t, remove_stopwords=remove_stopwords, normalize=normalize) for t in texts]

if __name__ == "__main__":
    sample = "I didn't like the product at all! It won't turn on and customer service was not helpful."
    print("Original: ", sample)
    print("Cleaned:  ", clean_text(sample))
