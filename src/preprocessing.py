import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words("english")) | {"rt"}

def nettoyer_tweet(tweet):
    tweet = tweet.replace("’", "'")
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)  # 🆕 supprime les handles (mention @xxx)
    tweet = re.sub(r"[^\w\s.,!?'-]", "", tweet)  # supprime emojis
    tweet = re.sub(r"\.{2,}", " ", tweet)  # supprime "..."

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)

    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in custom_stopwords and word not in string.punctuation
    ]
    return " ".join(clean_tokens)
