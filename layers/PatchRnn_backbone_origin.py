__all__ = ['PatchTST_backbone']

# Cell
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
import math
# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.PatchTST_layers import positional_encoding_3D, RNNAttentionScore,RelativePositionBias
from layers.RevIN import RevIN
from layers.llama import FeedForward_x,RMSNorm,precompute_freqs_cis,apply_rotary_emb
from einops import rearrange, repeat
# Cell
class PatchRnn_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: bool = False, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, pe_3d=False, gau=0, LEB=0, reluSquared=0, ffn=1, resi_dual=0, rnn_matrix=0,qkv_bias=1,gb=0,ts_dim=7,
                 **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        # if self.revin: self.revin_layer = DishTS('standard', n_series=c_in, seq_len=context_window)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, pe_3d=pe_3d, gau=gau, LEB=LEB
                                    , reluSquared=reluSquared, ffn=ffn, resi_dual=resi_dual, norm=norm,
                                    rnn_matrix=rnn_matrix, qkv_bias=qkv_bias,gb=gb,ts_dim=ts_dim,**kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.rope = RelativePositionBias(n_heads=n_heads)

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, z):  # z: [bs x n_vars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            # z,_ = self.revin_layer(z, mode='forward')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x n_vars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x n_vars x patch_len x patch_num]

        # model
        z, weight_u = self.backbone(z)  # z: [bs x n_vars x d_model x patch_num]
        z = self.head(z)  # z: [bs x n_vars x target_window]


        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            # z = self.revin_layer(z, mode='inverse')
            z = z.permute(0, 2, 1)
        return z, weight_u

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

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
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class LEMblock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x) -> Tensor:
        h = F.relu(self.linear_1(x))  # [B,L,in_dim]-> B,L,hidden_dim  这里可以 relu在linear之前
        h = self.dropout(self.linear_2(h))
        res = self.linear_res(x)
        out = self.layernorm(h + res)
        return out

class LEMencoder(nn.Module):
    def __init__(self, layer_num, in_dim, hidden_dim, out_dim,dropout_rate):
        super().__init__()
        self.encoder_layer_num = layer_num-1
        self.feature_projection = LEMblock(in_dim, hidden_dim, out_dim,dropout_rate)
        self.encoder_layers = nn.ModuleList([
            LEMblock(out_dim, out_dim, out_dim,dropout_rate) for _ in range (layer_num-1)
        ])
    def forward(self, x) -> Tensor:
        x =self.feature_projection(x)
        for i in range(self.encoder_layer_num ):
            x = self.encoder_layers[i](x)  # [B*N,hidden_dim] -> [B*N,hidden_dim]
        return x

class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=False, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, pe_3d=False, gau=0, LEB=0, reluSquared=0, ffn=1, resi_dual=0,
                 rnn_matrix=0,qkv_bias=1, gb=0,ts_dim=7,**kwargs):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.attn_mask = attn_mask
        # Input encoding
        q_len = patch_num
        if LEB:
            self.W_P = LEMencoder(1,patch_len,d_model * 2,d_model,dropout)
        else:
            self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        self.seq_len = q_len

        self.pe_3d = pe_3d
        # Positional encoding
        if self.pe_3d:
            self.W_pos = positional_encoding_3D(pe, learn_pe, c_in, q_len, d_model)
        else:
            self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.RNNAttentionScore = RNNAttentionScore(q_len=patch_num)
        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, self.RNNAttentionScore, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                  norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn, gau=gau, reluSquared=reluSquared, ffn=ffn, resi_dual=resi_dual,
                                  rnn_matrix=rnn_matrix,qkv_bias=qkv_bias,gb=gb,ts_dim=ts_dim)

        self.freqs_cis = precompute_freqs_cis(
            d_model // n_heads, max_seq_len * 2
        )

    def forward(self, x) -> Tensor:  # x: [bs x n_vars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x n_vars x patch_num x patch_len]
        x = self.W_P(x)
        # x: [bs x n_vars x patch_num x d_model]
        freqs_cis=None
        if self.pe_3d==1:
            x = x + self.W_pos  # 1.2950
            # print('w_pos:',self.W_pos[0][0][0].item())
            u = torch.reshape(x, (
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * n_vars x patch_num x d_model]
            u = self.dropout(u)
        elif self.pe_3d==2:
            u = torch.reshape(x, (
                x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * n_vars x patch_num x d_model]
            self.freqs_cis = self.freqs_cis.to(x.device)
            freqs_cis = self.freqs_cis[0: 0 + x.shape[2]]
        else:
            u = torch.reshape(x, (
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * n_vars x patch_num x d_model]
            u = self.dropout(u + self.W_pos)  # u: [bs * n_vars x patch_num x d_model]   w_pos  patch_num+d_model


        # Encoder
        if self.attn_mask:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num, device=x.device)
            z, weight_u = self.encoder(u, attn_mask=attn_mask,freqs_cis=freqs_cis)  # z: [bs * n_vars x patch_num x d_model]
        else:
            z, weight_u = self.encoder(u,freqs_cis=freqs_cis)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x n_vars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x n_vars x d_model x patch_num]

        return z, weight_u


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, RNNAttentionScore=None, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False, gau=0, reluSquared=0, ffn=1,
                 resi_dual=0, rnn_matrix=0,qkv_bias=1,gb=0,ts_dim=7):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, RNNAttentionScore=RNNAttentionScore,
                                                     n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn, gau=gau, ffn=ffn
                                                     , reluSquared=reluSquared, resi_dual=resi_dual,
                                                     rnn_matrix=rnn_matrix,qkv_bias=qkv_bias,gb=gb,ts_dim=ts_dim) for i in range(n_layers)])
        self.res_attention = res_attention
        self.resi_dual = resi_dual

        if "batch" in norm.lower():
            self.norm_final = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_final = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,freqs_cis:Optional[Tensor] = None):
        output = src
        scores = None
        preNorm_src = src
        if self.res_attention:
            if self.resi_dual:
                for mod in self.layers: preNorm_src, output, scores, weight_u = mod(output, preNorm_src=preNorm_src,
                                                                                    prev=scores,
                                                                                    key_padding_mask=key_padding_mask,
                                                                                    attn_mask=attn_mask,
                                                                                    freqs_cis=freqs_cis)

                output = output + self.norm_final(preNorm_src)  # finally norm
                return output, weight_u
            else:
                for mod in self.layers: output, scores, weight_u = mod(output, prev=scores,
                                                                       key_padding_mask=key_padding_mask,
                                                                       attn_mask=attn_mask,
                                                                       freqs_cis=freqs_cis)
                return output, weight_u

        else:
            if self.resi_dual:
                for mod in self.layers: preNorm_src, output, scores, weight_u = mod(output, preNorm_src=preNorm_src,
                                                                                    key_padding_mask=key_padding_mask,
                                                                                    attn_mask=attn_mask,
                                                                                    freqs_cis=freqs_cis)

                output = output + self.norm_final(preNorm_src)  # finally norm
                return output, weight_u
            else:
                for mod in self.layers: output, scores, weight_u = mod(output, key_padding_mask=key_padding_mask,
                                                                       attn_mask=attn_mask,
                                                                       freqs_cis=freqs_cis)
                return output, weight_u


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, RNNAttentionScore, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., ffn=1,
                 bias=True, activation="gelu", res_attention=False, pre_norm=False, gau=0
                 , reluSquared=0, resi_dual=0, rnn_matrix=0,qkv_bias=1,gb=0,ts_dim=7):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, RNNAttentionScore, d_k, d_v,
                                             attn_dropout=attn_dropout, proj_dropout=dropout,
                                             res_attention=res_attention, q_len=q_len, gau=gau
                                             , reluSquared=reluSquared, rnn_matrix=rnn_matrix,qkv_bias=qkv_bias,gb=gb,ts_dim=ts_dim)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        # self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
        #                         get_activation_fn(activation),
        #                         nn.Dropout(dropout),
        #                         nn.Linear(d_ff, d_model, bias=bias))

        self.ff = FeedForward_x(d_model,d_ff,multiple_of=256)

        self.ffn = ffn
        self.resi_dual = resi_dual
        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        elif "rms" in norm.lower():
            self.norm_ffn= RMSNorm(d_model, eps= 1e-5)
        else:
            self.norm_ffn = nn.LayerNorm(d_model)


        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, preNorm_src: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,freqs_cis: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores, weight_u = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                          attn_mask=attn_mask,freqs_cis=freqs_cis)
        else:
            src2, attn, scores, weight_u = self.self_attn(src, src, src, key_padding_mask=key_padding_mask,
                                                          attn_mask=attn_mask,freqs_cis=freqs_cis)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src2 = self.dropout_attn(src2)
        src = src + src2  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.resi_dual:
            preNorm_src = src2 + preNorm_src
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward

        if self.ffn:
            src2 = self.ff(src)
            ## Add & Norm
            src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout

            if not self.pre_norm:
                src = self.norm_ffn(src)

        if self.res_attention:
            if self.resi_dual:
                return src, preNorm_src, scores, weight_u
            else:
                return src, scores, weight_u
        else:
            if self.resi_dual:
                return src, preNorm_src, scores, weight_u
            else:
                return src, scores, weight_u


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, RNNAttentionScore, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, q_len=1, gau=0
                 , reluSquared=0, rnn_matrix=0,gb=0,ts_dim=7):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.gau = gau
        self.gb = gb

        if gb:
            self.G_V = nn.Sequential(
                nn.Linear(ts_dim*d_model,ts_dim*d_v* n_heads, bias=qkv_bias),
                # nn.SiLU()
            )
        else:
            self.G_V = nn.Sequential(
                nn.Linear(d_model, d_v * n_heads, bias=qkv_bias),
                # nn.SiLU()
            )


        self.ts_dim = ts_dim
        self.H_V = nn.Sequential(
            nn.Linear(ts_dim*d_model, d_v * n_heads, bias=qkv_bias),
            # nn.SiLU()
        )
        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, RNNAttentionScore,
                                                   attn_dropout=attn_dropout, res_attention=self.res_attention,
                                                   lsa=lsa, q_len=q_len, gau=gau, reluSquared=reluSquared,
                                                   rnn_matrix=rnn_matrix)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model, bias=qkv_bias), nn.Dropout(proj_dropout))
        self.move=MultiHeadEMA(d_model, ndim=2)

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,freqs_cis: Optional[Tensor] = None, adapter: Optional[Tensor]=None):

        bs ,seqlen,_= Q.shape



        if K is None: K = Q
        if V is None: V = Q
        Q = self.move(Q.transpose(0, 1), None, None).transpose(0, 1)
        K = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, seqlen, self.n_heads, self.d_k)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, seqlen, self.n_heads, self.d_k)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        if freqs_cis is not None:
            q_s, k_s = apply_rotary_emb(q_s, k_s, freqs_cis=freqs_cis)

        q_s =q_s.transpose(1,2)
        k_s = k_s.permute(0, 2, 3,1)



        v_s = self.W_V(V).view(bs, seqlen, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        if self.gb:
            h_1 =rearrange(Q, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', ts_d=self.ts_dim)
            # Flatten
            # covariates_flat = rearrange(covariates, 'b l r -> b (l r)')  # [B*N,L+H,r_hat] -> [B*N,(L+H)*r_hat]
            h_2 = h_1.permute(0,2,1,3) # b seg_num ts_d d_model
            h_3 =rearrange(h_2, 'b  seg_num ts_d d_model -> b seg_num (ts_d d_model)')
            h_4 =rearrange(Q, '(b ts_d) seg_num d_model -> b  seg_num (ts_d d_model)', ts_d=self.ts_dim)
            # print(h_3.eq(h_4))
            # h_s = self.H_V(rearrange(Q, '(b ts_d) seg_num d_model -> b  seg_num (ts_d d_model)', ts_d=self.ts_dim ))# rearrange(Q, '(b ts_d) seg_num d_model -> b  ts_d seg_num d_model)', ts_d=self.ts_dim ).permuate（0,2,1,3)
            # values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1) # .permuate(0,2,1,3)
            # h_s = repeat(h_s,'b seg_num dkHead -> (b ts_d) seg_num dkHead', ts_d=self.ts_dim).view(bs, seqlen, self.n_heads, self.d_v).transpose(1, 2)

            g_s = rearrange(self.G_V(h_3),'b  seg_num (ts_dim d_v n_heads)-> (b ts_dim) n_heads seg_num  d_v',ts_dim =self.ts_dim,n_heads=self.n_heads)

            # g_s= h_s+g_s
            # print(c.eq(b))
        else:
            g_s = self.G_V(Q).view(bs, seqlen, self.n_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores, weight_u = self.sdp_attn(q_s, k_s, v_s, g_s, prev=prev,
                                                                        key_padding_mask=key_padding_mask,
                                                                        attn_mask=attn_mask,adapter=adapter)
        else:
            output, attn_weights, attn_scores, weight_u = self.sdp_attn(q_s, k_s, v_s, g_s,
                                                                        key_padding_mask=key_padding_mask,
                                                                        attn_mask=attn_mask,adapter=adapter)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores, weight_u
        else:
            return output, attn_weights, attn_scores, weight_u


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, RNNAttentionScore, attn_dropout=0., res_attention=False, lsa=False, q_len=0,
                 gau=False, reluSquared=0, rnn_matrix=0):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.gau = gau
        self.RNNAttentionScore = RNNAttentionScore
        self.reluSquared = reluSquared
        self.rnn_matrix = rnn_matrix
        # self.gb=gb

    def forward(self, q: Tensor, k: Tensor, v: Tensor, g: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,adapter: Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]


        ########start########
        bsz, _, _,_ = q.shape
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)
        #########end###########

        # Add pre-softmax attention scores from the previous layer (optional)

        # 这里是原版残差 注意修改为 401行代码
        if prev is not None: attn_scores = attn_scores + prev

        if self.rnn_matrix:
            attn_scores,weight_u = self.RNNAttentionScore(attn_scores)


        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        if self.reluSquared:
            attn_weights = (F.relu(attn_scores)) ** 2
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
            # 注意381行
        weight_u = torch.zeros([1])


        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]
        ###########start##############
        if adapter is not None:
            adapter_scores = torch.matmul(q, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            adapter_scores = self.gate * F.softmax(adapter_scores.float(), dim=-1).type_as(q)
            output = output + torch.matmul(adapter_scores, adapter_v)
        ############end############
        if self.gau:
            # if self.gb:                 # bs, self.n_heads,seqlen,self.d_v
            #     # output = g * output  # batch seg_num head  ts_dim d_model  || batch*ts_dim head seg_num d_model
            #     output = g * rearrange(output, '(batch ts_dim) head seg_num d_model -> batch seg_num head  ts_dim d_model ',ts_dim=g.shape[3])
            #     output =rearrange(output,'batch seg_num head  ts_dim d_model  -> (batch ts_dim) head seg_num d_model ')
            # else:
                output = g * output

        if self.res_attention:
            return output, attn_weights, attn_scores, weight_u
        else:
            return output, attn_weights, attn_scores, weight_u




class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)   # 生成vandaer  系数
        kernel = (p * self.beta) * torch.exp(vander)   #   correspond Appendix  equation 26（公式26)   \boldsymbol{\phi} \odot \boldsymbol{\alpha} \odot \boldsymbol{\beta}
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega  # \alpha \odot \mathbf{x}_t

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = F.silu(out + residual)
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                x = F.pad(x, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = F.silu(out.permute(2, 0, 1) + residual)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.truncation)