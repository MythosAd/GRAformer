import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.P = 24 * 7;
        self.m = args.enc_in
        self.hidR = 100; # number of RNN hidden units
        self.hidC = 100; # number of CNN hidden units
        self.hidS = 5; # hidSkip
        self.Ck = 6;  # the kernel size of the CNN layers
        self.skip = 6;
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = 24 # The window size of the highway component
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=0.2);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = nn.Linear(args.seq_len-self.Ck+1, args.pred_len);
        # if (args.output_fun == 'sigmoid'):
        # self.output = F.sigmoid;
        # if (args.output_fun == 'tanh'):
        #     self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);

        # CNN
        # c = x.view(-1, 1, self.P, self.m);
        c = x.unsqueeze(dim=1)
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3); # batch channel order-feature

        # RNN
        r = c.permute(2, 0, 1).contiguous();  # order-feature batch channel
        out_state, r = self.GRU1(r);

        # r = self.dropout(torch.squeeze(r, 0));
        rn = self.dropout(out_state); # 改成多个
        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, -1);
            last = s.shape[-1]
            s = s.permute(2, 0, 3, 1).contiguous();
            s = s.view(self.pt, batch_size * last, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(-1, last* self.hidS);
            s = self.dropout(s);
            # r = torch.cat((r, s), 1);
            rn = torch.cat((rn, s.repeat(rn.shape[0],1,1)), 2);  # 改成多个

        res = self.linear1(rn);

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.m);
            res = res + z;

            res= res.permute(1,2,0).contiguous();

            res = self.output(res);
            res = res.permute(0,2,1);
        return res;



