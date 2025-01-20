# Q1) Illustrate  the functions for Reshaping, viewing, stacking, squeezing, and unsqueezing of tensors.

import torch
import math
# print("Hello")
# print(torch.cuda.is_available())
# print(torch.__version__)

MyTensor = torch.randint(1,100,(2,3))
print(MyTensor)
print(torch.reshape(MyTensor,(3,2)))
print(torch.stack((MyTensor,MyTensor)))

x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())
y = torch.squeeze(x)
print(y.size())

