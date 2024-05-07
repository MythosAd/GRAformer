import torch
import torch.nn as nn


class DeepVAR(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, predict_steps):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)

        self.predict_steps = predict_steps

    def forward(self, x, future_covariates):
        batch_size, seq_len, _ = x.shape

        h_lstm, _ = self.lstm(x)

        preds = []
        h = h_lstm[:, -1, :]
        for i in range(self.predict_steps):
            h = h.unsqueeze(1)

            mu = self.mu_layer(h)
            sigma = torch.exp(self.sigma_layer(h))

            epsilon = torch.randn(mu.size())
            pred = mu + sigma * epsilon

            h = torch.cat((h, pred), dim=1)

            preds.append(pred)

        preds = torch.cat(preds, dim=1)

        return preds


import torch
import torch.nn as nn
import torch.optim as optim


class DeepVAR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DeepVAR, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out


# Example usage
input_dim = 1  # Input dimension (e.g., number of features)
hidden_dim = 64  # Hidden layer dimension
num_layers = 2  # Number of LSTM layers
output_dim = 1  # Output dimension (e.g., for univariate time series)

model = DeepVAR(input_dim, hidden_dim, num_layers, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample input data (you would replace this with your own time series data)
input_data = torch.randn(100, 10, input_dim)  # (batch_size, sequence_length, input_dim)
target_data = torch.randn(100, 1)  # (batch_size, output_dim)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)

    # Compute loss
    loss = criterion(outputs, target_data)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# After training, you can use the model for forecasting by providing input sequences.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepAR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_samples):
        super(DeepAR, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer for the mean prediction
        self.linear_mean = nn.Linear(hidden_dim, output_dim)

        # Define the output layer for the variance prediction
        self.linear_variance = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Predict mean and variance
        mean = self.linear_mean(out[:, -1, :])
        variance = torch.exp(self.linear_variance(out[:, -1, :]))

        return mean, variance


# Example usage
input_dim = 1  # Input dimension (e.g., number of features)
hidden_dim = 64  # Hidden layer dimension
num_layers = 2  # Number of LSTM layers
output_dim = 1  # Output dimension (e.g., for univariate time series)
num_samples = 100  # Number of Monte Carlo samples for probabilistic forecasting

model = DeepAR(input_dim, hidden_dim, num_layers, output_dim, num_samples)

# Define loss function and optimizer (you may need a custom loss for probabilistic forecasting)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample input data (you would replace this with your own time series data)
input_data = torch.randn(100, 10, input_dim)  # (batch_size, sequence_length, input_dim)
target_data = torch.randn(100, 1)  # (batch_size, output_dim)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    mean, variance = model(input_data)

    # Compute loss (custom loss for probabilistic forecasting)
    loss = torch.mean(variance + (target_data - mean) ** 2)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# After training, you can use the model to generate probabilistic forecasts.