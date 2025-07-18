import nltk
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("twitter_samples")

import wandb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import twitter_samples

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import nettoyer_tweet

# Initialiser wandb
wandb.init(project="sentiment-analysis-nltk")

# Charger les données
pos = twitter_samples.strings("positive_tweets.json")
neg = twitter_samples.strings("negative_tweets.json")
tweets = pos + neg
labels = [1]*len(pos) + [0]*len(neg)

tweets_cleaned = [nettoyer_tweet(t) for t in tweets]

# Split
X_train, X_test, y_train, y_test = train_test_split(tweets_cleaned, labels, test_size=0.2, random_state=42)

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=1, max_iter=1000))
])
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Log wandb
wandb.log({
    "f1_score": report['weighted avg']['f1-score'],
    "precision": report['weighted avg']['precision'],
    "recall": report['weighted avg']['recall'],
    "accuracy": report['accuracy']
})

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma")
plt.title("Matrice de confusion finale")
plt.xlabel("Prédit")
plt.ylabel("Réel")
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()

# Sauvegarde
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logreg_tfidf_final.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
#wandb.save(model_path)