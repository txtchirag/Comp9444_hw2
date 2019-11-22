import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB



# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_size = 100
        self.num_layers = 2
        self.lstm = tnn.LSTM(input_size=50, hidden_size=100,dropout=0.2,num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = tnn.Linear(in_features=200, out_features=64)
        self.fc2 = tnn.Linear(in_features=64, out_features=1)
        self.dropout=tnn.Dropout(0.2)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size)
        padded_sequence = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        packed_output, _ = self.lstm(padded_sequence,
                                     (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out, _ = tnn.utils.rnn.pad_packed_sequence(packed_output)
        out = F.relu(self.fc1(out[1]))

        # out, _ = self.lstm(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        # out = F.relu(self.fc1(out[:, -1, :]))

        out = self.dropout(out)
        x = self.fc2(out)  # Linear(1)
        x = x.view(-1)  # flatten
        #print(x.shape)
        return x


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(15):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()