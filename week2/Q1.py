import torch

# Define variables a and b
a = torch.tensor(1.0, requires_grad=True)  # Example value for a
b = torch.tensor(2.0, requires_grad=True)  # Example value for b

# Define intermediate computations
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y

# Compute the gradient of z with respect to a
z.backward()

# Print the result
print(f"Gradient dz/da (using PyTorch): {a.grad.item()}")
print(f"Gradient dz/db (using PyTorch): {b.grad.item()}")
