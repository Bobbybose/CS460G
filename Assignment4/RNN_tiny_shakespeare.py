# Author: Bobby Bose
# Assignment 3: Text Generation with RNNs

import numpy as np
import tensorflow as tf


BATCH_SIZE = 64

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
class RNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, num_layers):
        super().__init__(self)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        # Using a GRU for this model
        self.gru = tf.keras.layers.GRU(num_layers, return_state=True, return_sequences=True)

    def call(self, x, training=False, return_state=False, states=None):
        x = self.embedding(x, training=training)

        if states is None:
            states = self.gru.get_initial_state(x)

        x, states = self.gru(x, training=training, initial_state=states)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def main():
    print(tf.__version__)
    # Reading in training data
    tiny_shakespeare = open('tiny-shakespeare.txt', 'rb').read().decode(encoding='utf-8')

    # Making a list of characters in training data
    characters = set(''.join(tiny_shakespeare))

    # Mapping each unique character to an integer
    charToInt = tf.keras.layers.StringLookup(vocabulary = list(characters))

    # Mapping each integer to a character
    intToChar = tf.keras.layers.StringLookup(vocabulary=charToInt.get_vocabulary(), invert=True)

    # Converting input to indices
    sentences_indices = charToInt(tf.strings.unicode_split(tiny_shakespeare, 'UTF-8'))
    sentences_indices = tf.data.Dataset.from_tensor_slices(sentences_indices)

    # Splitting input into examples of length 100
    example_length = 100
    num_examples = len(tiny_shakespeare)//(example_length+1)
    examples = sentences_indices.batch(example_length+1, drop_remainder=True)

    # Creating input and output sequences
    training_data = examples.map(create_input_output_examples)

    # Creating batches
    training_data = training_data.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # Initializing the GRU
    vocab_size = len(characters)
    model = RNN(vocab_size+1, 256, 200)

    # Initializing loss function
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Initializing optimizer
    model.compile(optimizer='adam', loss=loss)

    # Training the RNN
    model.fit(training_data, epochs=20)

    states = None
    curr_prediction = tf.constant(['ROMEO:'])
    full_prediction = [curr_prediction]
    for i in range(100):
        curr_prediction, states = predict(curr_prediction, charToInt, intToChar, model, states)
        full_prediction.append(curr_prediction)
    
    full_prediction = tf.strings.join(full_prediction)
    print(full_prediction[0].numpy().decode('utf-8'), '\n\n' + '_'*80)



def create_input_output_examples(example):
    return example[:-1], example[1:]


def predict(input, charToInt, intToChar, model, states=None):
    #Modifying input
    input = charToInt(tf.strings.unicode_split(input, 'UTF-8')).to_tensor()

    # Obtaining a prediction
    prediction, RNNstates = model(input, return_state=True, states=states)
    
    # Modifying prediction
    prediction = tf.random.categorical(prediction[:, -1, :], 1)
    prediction = intToChar(tf.squeeze(prediction, axis=-1))

    return prediction, states





main()