# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# https://github.com/facebookresearch/llama/blob/main/llama/model.py


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class PatchEmbedding(nn.Module):
    def __init__(self, configs):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(configs.patch_len,
                                         configs.d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        # self.value_embedding = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model,
        #                            kernel_size=3, padding=configs.stride, padding_mode='circular', bias=False)

        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        # padding 8           ( n+stride - self.patch_len)/self.stride   96+8 -16  88/8=11   11+1 =12
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [bs x n_vars x patch_num x patch_len]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # x: [(bs x n_vars) x patch_num x patch_len]
        # Input encoding
        # x = x.permute(0, 1, 3, 2)  # x: [bs x n_vars x patch_len x patch_num]
        x = self.value_embedding(x)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.individual = 0
        self.n_vars = configs.enc_in
        self.patch_num = int((configs.seq_len) / configs.stride)
        self.head_nf = configs.d_model * self.patch_num
        self.head_drop = 0
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(self.head_nf, configs.pred_len))
                self.dropouts.append(nn.Dropout(self.head_drop))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.head_nf, configs.pred_len)
            self.dropout = nn.Dropout(self.head_drop)

    def forward(self, x):  # x: [bs x n_vars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x n_vars x target_window]
        else:
            x = self.flatten(x)  # x: [bs x n_vars x d_model x patch_num]
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)



class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of  # multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Model(nn.Module):
    # def __init__(self, params: ModelArgs):
    def __init__(self, configs):
        super().__init__()
        self.params = configs
        # self.vocab_size = params.vocab_size
        # self.n_layers = params.n_layers
        self.n_layers = self.params.e_layers
        self.params.n_layers = self.params.e_layers
        self.params.dim = self.params.d_model
        self.pred_len=self.params.pred_len
        # self.tok_embeddings = ParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.params.multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        self.params.norm_eps: float = 1e-5

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.params.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.params))

        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = nn.Linear(self.params.dim, self.params.dim, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
            self.params.dim // self.params.n_heads, 1024 * 2
        )
        self.patch_embedding = PatchEmbedding(configs)
        self.flatten_Head = Flatten_Head(configs)

        self.relation_encoder = 1
        if self.relation_encoder == 1:
            self.relation_layers = torch.nn.ModuleList()
            for layer_id in range(self.params.n_layers):
                self.relation_layers.append(TransformerBlock(layer_id, self.params))

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):  # , start_pos: int
        start_pos = 0

        # _bsz, seqlen = tokens.shape
        # h = self.tok_embeddings(tokens)
        # do normlize
        # tokens.shape batch  step n_var
        means = tokens.mean(1, keepdim=True).detach()
        x_enc = tokens - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching
        x = x_enc.permute(0, 2, 1)  #   batch  step n_var -->batch n_vars step
        n_vars = x.shape[1]
        x = self.patch_embedding(x)  # x: [bs x n_vars x patch_num x d_model]
        patch_num = x.shape[2]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        self.freqs_cis = self.freqs_cis.to(x.device)

        # time-coder
        if self.relation_encoder == 1:
            h_relation = rearrange(x, 'bs n_vars patch_num d_model -> (bs patch_num) n_vars d_model')

            for layer in self.relation_layers:
                h_relation = layer(h_relation, start_pos, None, None)
            h_relation = self.norm(h_relation)  # bs*n_vars patch_num d_model
            h = rearrange(h_relation, '(bs patch_num) n_vars d_model -> (bs n_vars) patch_num d_model',
                          patch_num=patch_num)
        else:
            h = rearrange(x, 'bs n_vars patch_num d_model -> (bs n_vars) patch_num d_model', n_vars=n_vars)

        seqlen = h.shape[1]  # x: [bs x n_vars x patch_num x patch_len]
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        # time-coder
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)  # bs*n_vars patch_num d_model

        # output = self.output(h[:, -1, :])  # only compute last logits
        # return output.float()

        z = rearrange(h, '(bs n_vars) patch_num d_model -> bs n_vars patch_num d_model', n_vars=n_vars)
        z = self.flatten_Head(z)  # z: [bs x n_vars x predict_step]
        dec_out = z.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out  #
