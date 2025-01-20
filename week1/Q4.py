import numpy
import torch

np = numpy.arange(10)
print(np)
reshaped= np.reshape(2,5)
print(reshaped)

pytorch_tensor = torch.from_numpy(reshaped)
print(pytorch_tensor)
