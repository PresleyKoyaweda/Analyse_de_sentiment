import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

import nltk

for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# 📌 Initialisations
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words("english")) | {"rt"}

def nettoyer_tweet(tweet):
    """
    Nettoie un tweet en supprimant les mentions, URLs, emojis, stopwords et ponctuation.
    """
    tweet = tweet.replace("’", "'")
    tweet = re.sub(r"http\S+", "", tweet)                 # Supprime les URLs
    tweet = re.sub(r"#", "", tweet)                       # Supprime les hashtags (#)
    tweet = re.sub(r"@\w+", "", tweet)                    # Supprime les handles @xxx
    tweet = re.sub(r"[^\w\s.,!?'\-]", "", tweet)          # Supprime les emojis et symboles
    tweet = re.sub(r"\.{2,}", " ", tweet)                 # Remplace "..." par un espace

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)

    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in custom_stopwords and word not in string.punctuation
    ]

    return " ".join(clean_tokens)
