'''
Backpropagation is the process by which a neural network adjusts its internal parameters based on the difference between 
its predictions and the actual outcomes, enabling it to learn and improve over time.
'''

import torch
import numpy as np

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w*x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)
