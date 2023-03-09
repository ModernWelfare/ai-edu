# Import necessary libraries
from HelperClass.DataReader_1_0 import *
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Set filename for data
file_name = "ch04.npz"

# Define neural network model class


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


# Main function
if __name__ == '__main__':
    # Set maximum number of epochs to train for
    max_epoch = 500
    # Create data reader object and load data from file
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()

    # Set input size
    NUM_INPUT = 1
    # Get input and output data in numpy array form
    XTrain, YTrain = sdr.XTrain, sdr.YTrain
    # Create PyTorch dataset from numpy arrays
    torch_dataset = TensorDataset(
        torch.FloatTensor(XTrain), torch.FloatTensor(YTrain))

    # Create PyTorch data loader from dataset
    train_loader = DataLoader(
        dataset=torch_dataset,
        batch_size=32,
        shuffle=True,
    )

    # Use Mean Squared Error loss function because we are doing regression and not classification.
    # If we were doing classification, we would use Cross Entropy loss function.
    loss_func = nn.MSELoss()
    model = Model(NUM_INPUT)
    # Define optimizer for model parameters
    optimizer = Adam(model.parameters(), lr=1e-2)

    # Initialize list to store mean loss at every epoch
    e_loss = []
    # Loop through each epoch
    for epoch in range(max_epoch):
        # Initialize list to store mean loss at every batch
        b_loss = []
        # Loop through each batch in the data loader
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # Zero the gradients of the model parameters
            optimizer.zero_grad()
            # Forward pass through the model to get predictions
            pred = model(batch_x)
            # Calculate loss between predictions and true values
            # This is the averaged loss for the batch, not the sum
            loss = loss_func(pred, batch_y)
            # Store loss value for the batch
            b_loss.append(loss.cpu().data.numpy())
            # Backward pass to calculate gradients
            loss.backward()
            # Update model parameters using optimizer
            optimizer.step()
            b_loss.append(loss.cpu().data.numpy())
        # Store mean loss for the epoch
        e_loss.append(np.mean(b_loss))
        # Print mean loss for the epoch every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {np.mean(b_loss):.5f}")
    # Plot mean loss versus epoch
    plt.plot(e_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.show()
