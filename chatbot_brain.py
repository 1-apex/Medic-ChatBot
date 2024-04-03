import nltk
import re

from sklearn.model_selection import train_test_split

from data_preprocessing import read_file
from wiki_scrape import scrape_wiki
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
import warnings
warnings.filterwarnings('ignore')

medicine_list = ["crocin", "ibuprofen", "paracetamol", "aspirin", "amoxicillin", "benadryl", "loratadine"]
medicine_pattern = r'\b(?:' + '|'.join(map(re.escape, medicine_list)) + r')\b'

med_data = ["crocin"]


def create_rnn_model(vocab_size, max_sequence_length, num_classes):
    model = Sequential()

    # Embedding layer to learn word embeddings
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length))

    # Simple RNN layer with 128 units
    model.add(SimpleRNN(units=128))

    # Output layer with softmax activation for classification
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


def brain_response(user_response):
    bot_response = ''

    matches = re.findall(medicine_pattern, user_response.lower())
    flag = 1

    for match in matches:
        if match not in med_data:
            tokens = nltk.word_tokenize(match)
            tags = nltk.pos_tag(tokens)
            keyword = [word for word, pos in tags if pos.startswith('NN')]
            print("Keywords : ", keyword)
            for word in keyword:
                scrape_wiki(word)
            med_data.append(match)
            flag = 0
            return "------ New Data has been added to model!! --------"

    if not flag:
        return

    # Read and preprocess data
    padded_sequences, word_index, encoded_text = read_file()

    vocab_size = len(word_index)
    max_sequence_length = max(len(seq) for seq in padded_sequences)

    # Create the RNN model
    rnn_model = create_rnn_model(vocab_size, max_sequence_length, len(encoded_text))

    # Compile the model
    rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the summary of the model
    rnn_model.summary()

    # return bot_response
    return bot_response


brain_response("tell me about crocin")