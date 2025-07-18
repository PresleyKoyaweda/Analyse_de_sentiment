import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

lemmatizer = WordNetLemmatizer()
stopwords_english = stopwords.words("english")
custom_stopwords = set(stopwords.words("english")) | {"rt"}

def nettoyer_tweet(tweet):
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in custom_stopwords and word not in string.punctuation
    ]
    return " ".join(clean_tokens)