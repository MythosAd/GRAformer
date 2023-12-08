
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class normal_flow_layer(nn.Module):
    def __init__(self, d_model, c_out, out_len):
        super(normal_flow_layer, self).__init__()
        self.pred_len = out_len
        self.conv = nn.Sequential(nn.Conv1d(c_out, c_out, 2),
                                  nn.ReLU())
        self.mu = nn.Linear(d_model + out_len, out_len)
        self.sigma = nn.Linear(d_model + out_len, out_len)

    def forward(self, input_data, sample):
        h = self.conv(input_data)
        # h = h.squeeze()
        h = torch.cat((h[:, :, 0:1], h), 2)
        mu = self.mu(h)
        sigma = self.sigma(h)
        sample = mu + torch.exp(.5 * sigma) * sample * 0.1
        h = torch.cat((h[:, :, 0:h.shape[2] - self.pred_len], sample), 2)
        output = input_data + 0.1 * h
        return output, sample, sigma


class NormalizingFlow(nn.Module):


    def __init__(self, enc_in,pred_len,dropout,nflow_dim,nflow_layers):
        super(NormalizingFlow, self).__init__()

        # norm-flow
        self.d_ff = nflow_dim # 16
        self.LSTM = nn.GRU(input_size=enc_in, hidden_size=self.d_ff, num_layers=1, batch_first=True)

        self.enc_fix = nn.Linear(1, enc_in)

        self.distribution_dec_mu = nn.Linear(self.d_ff, pred_len)

        self.distribution_dec_presigma = nn.Linear(self.d_ff, pred_len)
        self.distribution_enc_mu = nn.Linear(self.d_ff, pred_len)
        self.distribution_enc_presigma = nn.Linear(self.d_ff,pred_len)
        self.distribution_sigma = nn.Softplus()

        self.dropout = nn.Dropout(dropout)
        self.normal_layers = nflow_layers
        self.normal_flow = nn.ModuleList(  # moduleList needs implementing forward function, sequential did not
            # [normal_flow_layer(self.d_model, self.c_out, self.pred_len) for l in range(self.normal_layers)])
            [normal_flow_layer(self.d_ff, enc_in, pred_len) for l in range(self.normal_layers)])


    def forward(self,x_enc):
        ####################### norm - flow ####
        _, hidden = self.LSTM(x_enc.permute(0, 2, 1))
        enc_hidden_permute = torch.permute(hidden, (1, 2, 0))
        enc_hidden_permute = torch.permute(self.enc_fix(enc_hidden_permute),
                                           (0, 2, 1))  # batch  从 d_model --> enc_out , d_ff
        # enc_hidden_permute =self.Linear_xl(rearrange(enc_out.permute(0, 1, 3, 2),'b dim segment_num d_model -> b (dim segment_num) d_model').permute(0,2,1)).permute(0,2,1) #y1 = y1.permute(1, 0, 2)  # batch dim  d_model
        enc_mu = self.distribution_enc_mu(enc_hidden_permute)
        enc_pre_sigma = self.distribution_enc_presigma(enc_hidden_permute)
        enc_sigma = self.distribution_sigma(enc_pre_sigma)

        dist = Normal(0, 1)
        eps = dist.sample(sample_shape=enc_mu.shape).cuda()  # batch dim pred_len
        sample = enc_mu + torch.exp(.5 * enc_sigma) * eps
        # sample = dec_mu + torch.exp(.5 * dec_sigma) * sample * 0.1

        h = torch.cat((enc_hidden_permute, sample), 2)  # batch dim   x+x
        for flow in self.normal_flow:
            h, sample, sigma = flow(h, sample)
        sample = torch.permute(sample, (0, 2, 1))

        return sample
        ################ end