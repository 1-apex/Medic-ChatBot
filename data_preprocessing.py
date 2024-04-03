import nltk
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re


def read_file():
    f = open('data.txt', 'r', errors='ignore')
    raw_doc = f.read()

    raw_doc = raw_doc.lower()

    # Use re.sub() to replace all occurrences of the [_] with an empty string
    raw_doc = re.sub(r"\[.*?\]", "", raw_doc)

    f = open('data.txt', 'w', errors='ignore')
    f.write(raw_doc)

    sentence_tokens = nltk.sent_tokenize(raw_doc)
    word_tokens = nltk.word_tokenize(raw_doc)

    # Removing stopwords: Remove common words that do not carry much meaning
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_tokens if word not in stop_words]

    # Remove punctuation and non-alphanumeric characters
    filtered_words = [word for word in filtered_words if word.isalnum()]

    # Tokenize and pad sequences
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(raw_doc)

    sequences = tokenizer.texts_to_sequences(sentence_tokens)
    padded_sequences = pad_sequences(sequences, padding='post')

    # Encoding: Map each word to a unique integer index
    word_counts = Counter(filtered_words)
    word_index = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
    word_index['OOV'] = len(word_index) + 1

    # Encode the text using the word index
    encoded_text = [word_index[word] for word in filtered_words]

    return [padded_sequences, word_index, encoded_text]


def lem_tokens(tokens):
    lem = nltk.stem.WordNetLemmatizer()
    return [lem.lemmatize(token) for token in tokens]


def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower()))

read_file()