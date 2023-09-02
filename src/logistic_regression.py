# This file is the same as linear_regression.py except we use the sigmoid function at the end to convert the output to a probability
# the logisitc regression model is used for binary classification problems.
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) # split data into train and test sets

sc = StandardScaler() # scale our data
X_train = sc.fit_transform(X_train) # fit and transform the training data
X_test = sc.transform(X_test) # transform the test data
X_train = torch.from_numpy(X_train.astype(np.float32)) # convert numpy array to torch tensor
X_test = torch.from_numpy(X_test.astype(np.float32)) # convert numpy array to torch tensor
y_train = torch.from_numpy(y_train.astype(np.float32)) # convert numpy array to torch tensor
y_test = torch.from_numpy(y_test.astype(np.float32)) # convert numpy array to torch tensor

y_train = y_train.view(y_train.shape[0], 1) # reshape y_train to be a column vector
y_test = y_test.view(y_test.shape[0], 1) # reshape y_test to be a column vector


# 1) model ,  f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__() # inherit from nn.Module
        self.linear = nn.Linear(n_input_features, 1) # define linear layer
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # apply sigmoid to the output of the linear layer
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss() # binary cross entropy loss. We use this in sigmoid classification problems because it is more numerically stable than MSE and it is mainly used for binary classification problems
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # stochastic gradient descent optimizer

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    

    # backward pass
    loss.backward()

    # updates
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) & 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test) # get the predictions
    y_predicted_cls = y_predicted.round() # round the output to 0 or 1
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) # calculate the accuracy. this equatoin is simply the number of correct predictions divided by the total number of predictions
    print(f'accuracy = {acc:.4f}') # print the accuracy





