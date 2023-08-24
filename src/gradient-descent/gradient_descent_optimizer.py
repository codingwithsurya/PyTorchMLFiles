'''
1) Design Model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
    
'''

# Import the necessary modules
import torch
import torch.nn as nn

# Define the input and target data for training
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # Input values
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # Corresponding target values

# Define a single test input data point
X_test = torch.tensor([5], dtype=torch.float32)

# Calculate the number of samples and features in the input data
n_samples, n_features = X.shape
print("Number of samples:", n_samples)
print("Number of features:", n_features)

# Set the input and output sizes for the linear model
input_size = n_features
output_size = n_features

# Create a linear model with a single weight and bias
# model = nn.Linear(input_size, output_size)

# Custom Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers 
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)
    

# Print the model's prediction using the test input before any training
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Set hyperparameters for training
learning_rate = 0.01  # Rate at which the model learns from the data
n_iters = 100  # Number of training iterations

# Define the loss function for training (Mean Squared Error)
loss = nn.MSELoss()

# Define the optimizer (Stochastic Gradient Descent) for updating model parameters
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop that iterates through multiple epochs
for epoch in range(n_iters):
    # Perform a forward pass to predict output using the current model
    y_pred = model(X)
    
    # Compute the Mean Squared Error (MSE) loss between predicted and actual target values
    l = loss(Y, y_pred)
    
    # Compute gradients of the loss with respect to model parameters
    l.backward()
    
    # Update model parameters (weights and biases) using the optimizer
    optimizer.step()
    
    # Zero out the gradients to prevent accumulation in the next iteration
    optimizer.zero_grad()

    # Print progress every 10 epochs, showing model weight and loss
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: weight = {w[0][0].item():.3f}, loss = {l:.8f}')

# Print the model's prediction using the test input after training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
