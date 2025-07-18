import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import nettoyer_tweet


import nltk
nltk.download("stopwords")
nltk.download("wordnet")

@pytest.mark.parametrize("tweet,expected", [
    ("This is a good day!", "good day"),  # remove stopwords and punctuation
    ("I love #Python and #AI!", "love python ai"),  # remove #
    ("@user123 This is amazing!", "amazing"),  # remove handle
    ("He’s running faster than ever!!!", "running faster ever"),  # strip punctuation, keep lemmatized word
    ("Working...working...WORKING!", "working working working"),  # repeated and case-handling
    ("RT @elonmusk: SpaceX is launching!", "spacex launching"),  # retweet and handle
    ("Why??? 😱😱", ""),  # emojis and punctuation
    ("Data, Data, and more Data.", "data data data"),  # repeat, punctuation
])
def test_nettoyer_tweet(tweet, expected):
    assert nettoyer_tweet(tweet) == expected
