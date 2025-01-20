import torch

# Data
x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

# Initial parameters
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Hyperparameters
learning_rate = 0.001
epochs = 2

# Gradient Descent for 2 epochs
for epoch in range(epochs):
    # Forward pass (model prediction)
    y_pred = w * x + b

    # Compute the loss (Mean Squared Error)
    loss = torch.mean((y - y_pred) ** 2)

    # Zero gradients from previous step
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # Backward pass (compute gradients)
    loss.backward()

    # Print the gradients
    print(f"Epoch {epoch + 1}:")
    print(f"w.grad = {w.grad.item():.4f}, b.grad = {b.grad.item():.4f}")

    # Update parameters using gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Print updated parameters
    print(f"Updated w = {w.item():.4f}, Updated b = {b.item():.4f}\n")

