'''
1) Design Model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
    
'''

import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) # generate dataset

X = torch.from_numpy(X_numpy.astype(np.float32)) # convert numpy array to torch tensor
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape y to be a column vector

n_samples, n_features = X.shape # get number of samples and number of features
 
# 1) model
model = nn.Linear(n_features, 1) # create a linear regression model

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # mean squared error loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # stochastic gradient descent optimizer

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X) # predict the output
    loss = criterion(y_predicted, y) # calculate loss

    #backward pass
    loss.backward() # perform backpropagation

    # update weights
    optimizer.step() # update weights
    optimizer.zero_grad() # zero out gradients

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy() # detach() detaches the output from the computationnal graph
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()



