# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, n_features: int, kernel_size: int = 7):
        super().__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)


class FeatureAttentionLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.window_size
        self.use_gatv2 = bool(use_gatv2)
        self.num_nodes = self.n_features
        self.use_bias = bool(use_bias)

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * self.window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = self.window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_features, self.n_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            a_input = self.leakyrelu(self.lin(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            wx = self.lin(x)
            a_input = self._make_attention_input(wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        if self.use_bias:
            e = e + self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))
        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        k = self.num_nodes
        blocks_repeating = v.repeat_interleave(k, dim=1)
        blocks_alternating = v.repeat(1, k, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        if self.use_gatv2:
            return combined.view(v.size(0), k, k, 2 * self.window_size)
        return combined.view(v.size(0), k, k, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.use_gatv2 = bool(use_gatv2)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.n_features
        self.num_nodes = self.window_size
        self.use_bias = bool(use_bias)

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * self.n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = self.n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.window_size, self.window_size))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            a_input = self.leakyrelu(self.lin(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            wx = self.lin(x)
            a_input = self._make_attention_input(wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        if self.use_bias:
            e = e + self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        return self.sigmoid(torch.matmul(attention, x))

    def _make_attention_input(self, v):
        k = self.num_nodes
        blocks_repeating = v.repeat_interleave(k, dim=1)
        blocks_alternating = v.repeat(1, k, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        if self.use_gatv2:
            return combined.view(v.size(0), k, k, 2 * self.n_features)
        return combined.view(v.size(0), k, k, 2 * self.embed_dim)


class GRULayer(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(
            in_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, x):
        out, h = self.gru(x)
        return out[-1, :, :], h[-1, :, :]


class RNNDecoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(
            in_dim,
            hid_dim,
            n_layers,
            batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h_end_rep = x.repeat_interleave(self.window_size, dim=1).view(
            x.size(0), self.window_size, -1
        )
        decoder_out = self.decoder(h_end_rep)
        return self.fc(decoder_out)


class ForecastingModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)


class MTADGATNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_dim: int,
        kernel_size: int = 7,
        feat_gat_embed_dim: int | None = None,
        time_gat_embed_dim: int | None = None,
        use_gatv2: bool = True,
        gru_n_layers: int = 1,
        gru_hid_dim: int = 150,
        forecast_n_layers: int = 3,
        forecast_hid_dim: int = 150,
        recon_n_layers: int = 1,
        recon_hid_dim: int = 150,
        dropout: float = 0.3,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            feat_gat_embed_dim,
            use_gatv2,
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            time_gat_embed_dim,
            use_gatv2,
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = ForecastingModel(
            gru_hid_dim,
            forecast_hid_dim,
            out_dim,
            forecast_n_layers,
            dropout,
        )
        self.recon_model = ReconstructionModel(
            window_size,
            gru_hid_dim,
            recon_hid_dim,
            out_dim,
            recon_n_layers,
            dropout,
        )

    def forward(self, x):
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)
        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)
        return predictions, recons


__all__ = [
    "ConvLayer",
    "FeatureAttentionLayer",
    "ForecastingModel",
    "GRULayer",
    "MTADGATNetwork",
    "ReconstructionModel",
    "TemporalAttentionLayer",
]
