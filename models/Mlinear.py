import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.Linear_ci = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_cd = nn.Linear(self.seq_len * self.enc_in, self.pred_len * self.c_out)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        weight_u = torch.rand([1])
        nn.init.constant_(weight_u, 0)
        self.weight_u = nn.Parameter(weight_u, requires_grad=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        x_ci = self.Linear_ci(x.permute(0, 2, 1)).permute(0, 2, 1)

        x_cd = self.Linear_cd(rearrange(x, 'b seq_len enc_in -> b (seq_len enc_in)'))
        x_cd = rearrange(x_cd, 'b (pred_len c_out) -> b pred_len c_out', pred_len=self.pred_len)
        weight = F.sigmoid(self.weight_u)
        out = weight * x_ci + (1 - weight) * x_cd
        dec_out = out + seq_last

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))



        return dec_out,weight  # [Batch, Output length, Channel]
