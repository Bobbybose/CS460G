# Author: Bobby Bose
# Assignment 3: Text Generation with RNNs
# Note: Based off of example by Dr. Brent Harrison showed during class

import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        # Storing network parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Creating RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Initializing hidden state
        hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size)

        # Running input through RNN
        output, hidden_state = self.rnn(x, hidden_state)

        # Reforming because not doing batches
        output = output.contiguous().view(-1, self.hidden_size)

        # Receiving output from fully connected layer
        output = self.fc(output)

        return output, hidden_state


def main():
    # Reading in training data
    tiny_shakespeare = open('tiny-shakespeare.txt', 'r')
    sentences = [line for line in tiny_shakespeare.readlines() if line.strip()]

    # Making a list of characters in training data
    characters = set(''.join(sentences))

    # Map index to character
    intToChar = dict(enumerate(characters))

    # Map character to index
    charToInt = {character: index for index, character in intToChar.items()}

    # Obtaining input and output sequences
    input_sequence = []
    target_sequence = []
    for i in range(len(sentences)):
        input_sequence.append(sentences[i][:-1])
        target_sequence.append(sentences[i][1:])

    # Replacing all characters with associated integer
    for i in range(len(sentences)):
        input_sequence[i] = [charToInt[character] for character in input_sequence[i]]
        target_sequence[i] = [charToInt[character] for character in target_sequence[i]]

    # Size of training vocabulary
    vocab_size = len(charToInt)

    # Initializing the RNN model
    model = RNN(vocab_size, vocab_size, 300, 1)

    # Initializing loss function
    loss = nn.CrossEntropyLoss()

    # Initializing optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Training the RNN
    for epoch in range(5):
        for i in range(len(input_sequence)):
            # Zeroing out gradients
            optimizer.zero_grad()
            
            # Creating a tensor for the input
            x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))

            # Sequence output for input
            y = torch.Tensor(target_sequence[i])

            # Running the RNN
            output, hidden = model(x)

            lossValue = loss(output, y.view(-1).long())
            lossValue.backward()
            optimizer.step()

        print("Loss: " + str(lossValue.item()))


def create_one_hot(sequence, vocab_size):
    encoding = np.zeros((1, len(sequence), vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    
    return encoding


main()