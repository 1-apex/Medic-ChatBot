import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentences, all_words):
    tokenized_sentences = [stem(w) for w in tokenized_sentences]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentences:
            bag[idx] = 1.0

    return bag

