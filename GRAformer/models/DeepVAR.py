# class Model(nn.Module):
#
#     # https://arxiv.org/abs/1910.03002
#     # These models have been described as VEC-LSTM in this paper: https://arxiv.org/abs/1910.03002
#     # High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes   multivariate
#     # https://readpaper.com/pdf-annotate/note?pdfId=4546283723980169217
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super(DeepVAR, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # LSTM layer
#         x, _ = self.lstm(x)
#
#         # Take the output of the last time step
#         x = x[:, -1, :]
#
#         # Fully connected layer
#         x = self.fc(x)
#         return x