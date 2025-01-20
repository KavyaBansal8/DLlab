import torch
import matplotlib.pyplot as plt


# Define the RegressionModel class
class RegressionModel:
    def __init__(self):
        # Initialize parameters w and b to 1
        self.w = torch.tensor(1.0, requires_grad=True)
        self.b = torch.tensor(1.0, requires_grad=True)

    # Forward pass to compute predicted values
    def forward(self, x):
        return self.w * x + self.b

    # Update the parameters using gradient descent
    def update(self, learning_rate):
        # Update parameters with gradient descent
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad

    # Reset gradients to zero
    def reset_grad(self):
        if self.w.grad is not None:
            self.w.grad.zero_()
        if self.b.grad is not None:
            self.b.grad.zero_()

    # Compute MSE loss between target and predicted values
    def criterion(self, y, yp):
        return torch.mean((y - yp) ** 2)


# Initialize the model
model = RegressionModel()

# Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

# Learning rate
learning_rate = 0.001

# Number of epochs
epochs = 10

# Store loss values for plotting
losses = []

# Training loop for 100 epochs
for epoch in range(epochs):
    # Forward pass: calculate predicted values
    y_pred = model.forward(x)

    # Compute the loss (Mean Squared Error)
    loss = model.criterion(y, y_pred)

    # Reset gradients before backward pass
    model.reset_grad()

    # Backward pass: compute gradients
    loss.backward()

    # Update parameters using gradient descent
    model.update(learning_rate)

    # Append the loss value for plotting
    losses.append(loss.item())

    # Print loss every 10 epochs for monitoring
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Plot the loss vs epoch
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()
