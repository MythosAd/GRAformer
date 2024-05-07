import torch
import numpy as np

class VARModelTorchGPU:
    def __init__(self, data, lags):
        self.data = data
        self.lags = lags
        self.n = data.shape[1]  # Number of variables

    def fit(self):
        # Preparing the lagged data matrix
        X, Y = self.prepare_data(self.data, self.lags)

        # Converting to PyTorch tensors and moving to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        Y_tensor = torch.tensor(Y, dtype=torch.float32).cuda()

        # Using PyTorch's linear regression (least squares) to fit the model
        # beta = (X'X)^(-1)X'Y
        XTX = X_tensor.T @ X_tensor
        XTY = X_tensor.T @ Y_tensor
        self.beta = torch.linalg.solve(XTX, XTY)

        return self.beta

    def predict(self,X ,steps):
        predictions = []


        for _ in range(steps):
            # Preparing the data for prediction
            #X = self.prepare_lagged_data(last_obs, self.lags)
            X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
            # Forecasting
            forecast = X_tensor.flatten() @ self.beta
            predictions.append(forecast.cpu().detach().numpy().flatten())
            # Update the last observations with the forecast

        return np.array(predictions)

    @staticmethod
    def prepare_data(data, lags):
        """Prepare data for VAR model."""
        X, Y = [], []
        for i in range(lags, len(data)):
            X.append(data[i-lags:i].flatten())
            Y.append(data[i])
        return np.array(X), np.array(Y)

    @staticmethod
    def prepare_lagged_data(data, lags):
        """Create a lagged data matrix for prediction."""
        return data[-lags:].flatten().reshape(1, -1)