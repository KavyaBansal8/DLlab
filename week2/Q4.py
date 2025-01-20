import torch
# Input
x = torch.tensor(2.0, requires_grad=True)  # Example value for x

# Define the function
f = torch.exp(-x**2 - 2*x - torch.sin(x))

# Backpropagation
f.backward()

# Gradient
print(f"Gradient df/dx (using PyTorch): {x.grad.item()}")
