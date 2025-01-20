import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Define the RegressionModel class by extending nn.Module
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Define parameters as torch tensors, with requires_grad=True to track gradients
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Implement the forward pass
        return self.w * x + self.b


# Define the custom Dataset class
class LinearRegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        # Return the length of the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return a tuple (x, y) for a given index
        return self.x_data[idx], self.y_data[idx]


# Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)  # Reshaping for batch processing
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

# Hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 5  # We will use the entire dataset as a single batch for simplicity

# Create the dataset and data loader
dataset = LinearRegressionDataset(x, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = RegressionModel()

# Use SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Store loss values for plotting
losses = []

# Training loop for 100 epochs
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in data_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: calculate predicted values
        y_pred = model(batch_x)

        # Compute the loss (Mean Squared Error)
        loss = nn.MSELoss()(y_pred, batch_y)

        # Backward pass: compute gradients
        loss.backward()

        # Update parameters using the optimizer
        optimizer.step()

        # Accumulate loss for the epoch
        epoch_loss += loss.item()

    # Append the average loss for the epoch
    losses.append(epoch_loss / len(data_loader))

    # Print loss every 10 epochs for monitoring
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

# Plot the loss vs epoch
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()

# Print the final learned parameters
print(f"Learned w: {model.w.item()}, b: {model.b.item()}")
