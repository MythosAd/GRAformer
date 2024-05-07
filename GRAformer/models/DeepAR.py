
import torch
import torch.nn as nn
import torch.distributions as td

# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
# https://arxiv.org/abs/1704.04110
# https://readpaper.com/pdf-annotate/note?pdfId=4557660417334190081&noteId=2049286360205840640
def gaussian_likelihood_loss(mu, sigma, target):
    dist = td.Normal(mu, sigma)
    return -dist.log_prob(target).mean()

class DeepAR(nn.Module):
    def __init__(self,args, input_size, hidden_size, num_layers):
        super(DeepAR, self).__init__()

        input_size = 10  # Number of features
        hidden_size = 50  # Number of features in the hidden state
        num_layers = 2  # Number of LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Output layers for mean and standard deviation
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_sigma = nn.Linear(hidden_size, 1)



    def forward(self, x,y):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output for the last time step

        # Predict mean and log variance (for stability)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))  # Ensure sigma is positive

        loss = gaussian_likelihood_loss(mu, sigma, y)

        return loss,mu