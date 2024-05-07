import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class GPT4TS(nn.Module):

    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        #  3  trend season residul
        self.prompt = self.prompt_token('zeros', True, 3, configs.d_model, token_num=configs.prompt_token).to(device=device)

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('.//gpt_model', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

            if configs.freeze and configs.pretrain:
                for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


            self.gpt2.to(device=device)
            # 思考下 这里打开train()是否对evaluate 有影响？
            self.gpt2.train()

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model).to(device=device)
        self.out_layer = nn.Linear(configs.d_model * (self.patch_num + configs.prompt_token), configs.pred_len).to(device=device)

        # for layer in (self.gpt2, self.in_layer, self.out_layer):
        #     layer.to(device=device)
        #     layer.train()
        #
        self.cnt = 0

        self.decomp = series_decomp(25)
        self.time = torch.arange(0, configs.seq_len, 1).to(device=device)

    # 定义 Loess 平滑函数
    def loess_smoothing(self, x, y, span=0.5):
        n = len(x)
        y_smoothed = torch.zeros_like(y)

        for i in range(n):
            dist = torch.abs(x - x[i])
            weights = torch.exp(-0.5 * (dist / span) ** 2)
            weights = weights / torch.sum(weights)
            y_smoothed[:, i, :] = torch.einsum('j,bjk->bk', weights, y)

        return y_smoothed

    def forward(self, x):

        B, L, M = x.shape

        trend, season_res = self.decomp(x)
        # 进行 Loess 平滑
        season = self.loess_smoothing(self.time, season_res)
        # 计算residual
        res = x - trend - season

        x = torch.concatenate((trend, season, res), dim=0)

        # norm
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'i j k -> (i k) j')
        x = self.padding_patch_layer(x.squeeze(1))
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        outputs = self.in_layer(x)
        # 间隔复制
        prompt = self.prompt.repeat_interleave(B,dim=0)
        outputs = torch.concatenate((prompt,outputs), dim=1)
        # self.prompt.unsqueeze(0).repeat(B, 1, 1, 1).reshape()
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        # torch.concatenate((trend, season, res), dim=0)
        trend, season, res = torch.split(outputs, [B, B, B])

        trend_y = self.out_layer(trend.reshape(B * M, -1))
        season_y = self.out_layer(season.reshape(B * M, -1))
        res_y = self.out_layer(res.reshape(B * M, -1))

        trend_y = rearrange(trend_y, '(b m) l -> b l m', b=B)
        season_y = rearrange(season_y, '(b m) l -> b l m', b=B)
        res_y = rearrange(res_y, '(b m) l -> b l m', b=B)

        outputs = torch.concatenate((trend_y, season_y, res_y), dim=0)
        # denorm
        outputs = outputs * stdev
        outputs = outputs + means

        trend_z, season_z, res_z = torch.split(outputs, [B, B, B])

        outputs_z = trend_z + season_z + res_z

        return outputs_z

    def saveimg(self, x, trend, season, res):
        # 设置子图布局
        fig, axs = plt.subplots(4, 1, figsize=(10, 18))

        # 绘制原始信号图
        axs[0].plot(self.time.cpu().numpy(), x[0].squeeze(dim=1).cpu().numpy(), label="Original Component",
                    color="black")
        axs[0].set_title("Original Component")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Value")

        # 绘制趋势项子图
        axs[1].plot(self.time.cpu().numpy(), trend[0].squeeze(dim=1).cpu().numpy(), label="Trend Component",
                    color="green")
        axs[1].set_title("Trend Component")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Value")

        # 绘制季节项子图
        axs[2].plot(self.time.cpu().numpy(), season[0].squeeze(dim=1).cpu().numpy(), label="Seasonal Component",
                    color="red")
        axs[2].set_title("Seasonal Component")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Value")

        # 绘制残差项子图
        axs[3].plot(self.time.cpu().numpy(), res[0].squeeze(dim=1).cpu().numpy(), label="Residual Component",
                    color="blue")
        axs[3].set_title("Residual Component")
        axs[3].set_xlabel("Time")
        axs[3].set_ylabel("Value")

        plt.tight_layout()
        plt.show()
        plt.savefig('season.jpg')

    def prompt_token(self, pe, learn_pe, n_vars, d_model, token_num=1):
        # Positional encoding
        if pe == None:
            prompt = torch.empty(
                (n_vars, 1, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
            nn.init.uniform_(prompt, -0.02, 0.02)
            learn_pe = False
        elif pe == 'zero':
            prompt = torch.empty((n_vars, token_num, 1))
            nn.init.uniform_(prompt, -0.02, 0.02)
        elif pe == 'zeros':
            prompt = torch.empty((n_vars, token_num, d_model))
            nn.init.uniform_(prompt, -0.02, 0.02)
        elif pe == 'normal' or pe == 'gauss':
            prompt = torch.zeros((n_vars, token_num, 1))
            torch.nn.init.normal_(prompt, mean=0.0, std=0.1)
        elif pe == 'uniform':
            prompt = torch.zeros((n_vars, token_num, 1))
            nn.init.uniform_(prompt, a=0.0, b=0.1)
        else:
            raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        return nn.Parameter(prompt, requires_grad=learn_pe)  # n_vars x 1 x d_model


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return moving_mean, res


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    defualt : moving_avg=25 stirde= 1
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
