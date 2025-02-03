import torch
import torch.nn as nn
import torch.nn.functional as F

# Input image (from Q1, expanded to 4D: [batch=1, channel=1, H=6, W=6])
image = torch.rand(1, 1, 6, 6)  # Shape: (1, 1, 6, 6)

# Method 1: Using nn.Conv2d (with out_channels=3)
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=3,
    kernel_size=3,
    bias=False  # Ignore bias as per the problem statement
)
out_nn = conv_layer(image)
print("Output shape (nn.Conv2d):", out_nn.shape)  # Output: (1, 3, 4, 4)

# Method 2: Using F.conv2d (equivalent implementation)
# Manually create the kernel with shape (out_channels=3, in_channels=1, 3, 3)
kernel = torch.rand(3, 1, 3, 3)  # Same shape as conv_layer.weight
out_functional = F.conv2d(image, kernel, stride=1, padding=0)
print("Output shape (F.conv2d):", out_functional.shape)  # Output: (1, 3, 4, 4)

# Verify equivalence (if using the same kernel weights)
conv_layer.weight.data = kernel  # Sync weights
out_nn_synced = conv_layer(image)
print("Outputs are equal:", torch.allclose(out_nn_synced, out_functional))  # True