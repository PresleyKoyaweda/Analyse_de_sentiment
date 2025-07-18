import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# 📥 Télécharger automatiquement les ressources NLTK si elles manquent
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# 🔧 Initialisation
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words("english")) | {"rt"}

# 🧼 Fonction de nettoyage de tweet
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
