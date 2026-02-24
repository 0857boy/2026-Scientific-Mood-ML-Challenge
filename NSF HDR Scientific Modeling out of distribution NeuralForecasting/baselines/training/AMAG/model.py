"""
AMAG (Adaptive Multi-scale Attention Graph) model architecture
for neural time-series forecasting.

This file contains the model definitions used during training.
The submission version in baselines/submissions/AMAG/model.py
is a self-contained copy with the ingestion wrapper.

Author: Joe Liao, National Central University (NCU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes along the time dimension (dim=1) for each batch, node,
    and feature independently.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1, self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * 1e-5)
        x = x * self.stdev
        x = x + self.mean
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.1, 0.1)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiScaleAttention(nn.Module):
    """Fuses local (depthwise conv) and global (multi-head attention) features."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(MultiScaleAttention, self).__init__()
        self.conv_local = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        res = x
        # Local convolution
        x_local = x.permute(0, 2, 1)
        x_local = F.gelu(self.conv_local(x_local))
        x_local = x_local.permute(0, 2, 1)
        # Global attention
        x_global, _ = self.attn(x, x, x)
        # Gated fusion
        concat = torch.cat([x_local, x_global], dim=-1)
        out = F.gelu(self.gate(concat))
        return self.norm(res + self.dropout(out))


class AdaptiveGraphLayer(nn.Module):
    """Learns a sparse adjacency matrix from node embeddings and performs
    sample-dependent graph message passing."""

    def __init__(self, num_nodes, d_model, sparsity_threshold=0.01):
        super(AdaptiveGraphLayer, self).__init__()
        self.num_nodes = num_nodes
        self.sparsity_threshold = sparsity_threshold
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 128))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 128))

        self.lin_add = nn.Linear(d_model, d_model)
        self.lin_mul = nn.Linear(d_model, d_model)

        # Sample-dependent adaptor
        self.adaptor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.lin_out = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def init_from_correlation(self, corr_matrix):
        """Initialize node embeddings from a pre-computed correlation matrix."""
        try:
            if hasattr(torch.linalg, 'svd'):
                U, S, Vh = torch.linalg.svd(
                    torch.tensor(corr_matrix).float(), full_matrices=False
                )
                V = Vh.mH
            else:
                U, S, V = torch.svd(torch.tensor(corr_matrix).float())

            k = min(128, S.size(0))
            U_k = U[:, :k]
            S_k = torch.sqrt(S[:k])
            V_k = V[:, :k]

            emb1 = U_k * S_k.unsqueeze(0)
            emb2 = V_k * S_k.unsqueeze(0)

            if k < 128:
                pad = 128 - k
                emb1 = F.pad(emb1, (0, pad))
                emb2 = F.pad(emb2, (0, pad))

            self.node_emb1.data.copy_(emb1)
            self.node_emb2.data.copy_(emb2)
        except Exception:
            pass

    def forward(self, x):
        B, T, N, D = x.shape

        raw_adj = torch.mm(self.node_emb1, self.node_emb2.t())
        mask = raw_adj > self.sparsity_threshold
        masked_adj = raw_adj.masked_fill(~mask, -1e9)
        adj = F.softmax(masked_adj, dim=-1)

        x_flat = x.reshape(-1, N, D)
        adj_expanded = adj.unsqueeze(0).expand(B * T, -1, -1)
        aggr = torch.bmm(adj_expanded, x_flat).reshape(B, T, N, D)

        ctx = x.mean(dim=1)
        gate = self.adaptor(ctx).unsqueeze(1)
        aggr = aggr * gate

        out_add = self.lin_add(aggr)
        out_mul = self.lin_mul(aggr) * x
        out = torch.cat([out_add, out_mul], dim=-1)
        out = self.lin_out(out)

        return self.norm(x + out)


class AMAG(nn.Module):
    """
    Adaptive Multi-scale Attention Graph (AMAG) model.

    Combines RevIN normalization, multi-scale temporal attention,
    and adaptive graph convolution for neural time-series forecasting.

    Args:
        monkey_name: 'affi' (239 nodes) or 'beignet' (87 nodes)
        d_model: hidden dimension
        n_heads: number of attention heads
        n_layers: number of stacked TE + SI layers
    """
    def __init__(self, monkey_name='affi', d_model=64, n_heads=4, n_layers=2):
        super(AMAG, self).__init__()

        if monkey_name == 'affi':
            self.num_nodes = 239
        elif monkey_name == 'beignet':
            self.num_nodes = 87
        else:
            raise ValueError(f"Unknown monkey: {monkey_name}")

        self.obs_len = 10
        self.pred_len = 10
        self.input_feats = 9

        self.revin = RevIN(self.input_feats, affine=False)
        self.embedding = nn.Linear(self.input_feats, d_model)
        self.pe = LearnablePositionalEncoding(d_model, max_len=20)

        self.te_layers = nn.ModuleList([
            MultiScaleAttention(d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        self.si_layers = nn.ModuleList([
            AdaptiveGraphLayer(self.num_nodes, d_model)
            for _ in range(n_layers)
        ])

        self.temporal_map = nn.Linear(self.obs_len, self.pred_len)
        self.out_head = nn.Linear(d_model, 1)

    def forward_core(self, x_norm):
        """Core prediction pipeline (normalized input -> normalized output)."""
        B, _, N, _ = x_norm.shape

        h = self.embedding(x_norm)
        h = h.permute(0, 2, 1, 3).reshape(B * N, 10, -1)
        h = self.pe(h)

        for te, si in zip(self.te_layers, self.si_layers):
            h = te(h)
            h = h.view(B, N, 10, -1).permute(0, 2, 1, 3)
            h = si(h)
            h = h.permute(0, 2, 1, 3).reshape(B * N, 10, -1)

        h_trans = h.permute(0, 2, 1)
        pred = self.temporal_map(h_trans).permute(0, 2, 1)
        out = self.out_head(pred)

        return out.view(B, N, 10).permute(0, 2, 1)

    def forward(self, x):
        """
        Full pipeline: (B, 20, N, F) -> RevIN -> Core -> Denorm -> (B, 20, N)
        """
        x_obs = x[:, :self.obs_len, :, :]
        x_norm = self.revin(x_obs, 'norm')
        out_norm = self.forward_core(x_norm)

        mean_0 = self.revin.mean[..., 0]
        stdev_0 = self.revin.stdev[..., 0]
        out_denorm = out_norm * stdev_0 + mean_0

        obs_vals = x_obs[..., 0]
        full_pred = torch.cat([obs_vals, out_denorm], dim=1)

        return full_pred
