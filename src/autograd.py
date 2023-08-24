'''
Autograd automatically computes how changing inputs affects function outputs, crucial for training models.
'''
import torch
import numpy as np

weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
