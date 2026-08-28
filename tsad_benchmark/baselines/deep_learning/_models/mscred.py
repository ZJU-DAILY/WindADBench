# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad2d(x: torch.Tensor, kernel_size: int, stride: int = 1) -> torch.Tensor:
   
    h, w = int(x.shape[-2]), int(x.shape[-1])
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    pad_h = max((out_h - 1) * stride + kernel_size - h, 0)
    pad_w = max((out_w - 1) * stride + kernel_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(
        x,
        [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ],
    )


class SameConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_same_pad2d(x, self.kernel_size, self.stride))


class ConvLSTMCell(nn.Module):
   

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            int(input_dim) + self.hidden_dim,
            4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            size = (x.size(0), self.hidden_dim, x.size(2), x.size(3))
            h = torch.zeros(size, dtype=x.dtype, device=x.device)
            c = torch.zeros(size, dtype=x.dtype, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(_same_pad2d(combined, self.kernel_size, stride=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class AttentionConvLSTM(nn.Module):
  

    def __init__(self, channels: int, step_max: int):
        super().__init__()
        self.step_max = int(step_max)
        self.cell = ConvLSTMCell(channels, channels, kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        state = None
        outputs = []
        for t in range(x.shape[1]):
            state = self.cell(x[:, t], state)
            outputs.append(state[0])
        stacked = torch.stack(outputs, dim=1)
        last = stacked[:, -1:].detach() if not self.training else stacked[:, -1:]
        scores = (stacked * last).flatten(start_dim=2).sum(dim=2)
        scores = scores / float(max(self.step_max, 1))
        weights = torch.softmax(scores, dim=1).view(x.shape[0], x.shape[1], 1, 1, 1)
        return torch.sum(stacked * weights, dim=1)


class MSCREDNetwork(nn.Module):
 
    def __init__(self, n_scales: int, step_max: int = 5):
        super().__init__()
        self.n_scales = int(n_scales)
        self.step_max = int(step_max)

        self.conv1 = SameConv2d(self.n_scales, 32, kernel_size=3, stride=1)
        self.conv2 = SameConv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = SameConv2d(64, 128, kernel_size=2, stride=2)
        self.conv4 = SameConv2d(128, 256, kernel_size=2, stride=2)

        self.lstm1 = AttentionConvLSTM(32, self.step_max)
        self.lstm2 = AttentionConvLSTM(64, self.step_max)
        self.lstm3 = AttentionConvLSTM(128, self.step_max)
        self.lstm4 = AttentionConvLSTM(256, self.step_max)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, self.n_scales, kernel_size=3, stride=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _encode_one(self, frame: torch.Tensor):
        conv1 = F.selu(self.conv1(frame))
        conv2 = F.selu(self.conv2(conv1))
        conv3 = F.selu(self.conv3(conv2))
        conv4 = F.selu(self.conv4(conv3))
        return conv1, conv2, conv3, conv4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, F, F)
        encoded = [self._encode_one(x[:, t]) for t in range(x.shape[1])]
        conv1_seq = torch.stack([item[0] for item in encoded], dim=1)
        conv2_seq = torch.stack([item[1] for item in encoded], dim=1)
        conv3_seq = torch.stack([item[2] for item in encoded], dim=1)
        conv4_seq = torch.stack([item[3] for item in encoded], dim=1)

        conv1_lstm = self.lstm1(conv1_seq)
        conv2_lstm = self.lstm2(conv2_seq)
        conv3_lstm = self.lstm3(conv3_seq)
        conv4_lstm = self.lstm4(conv4_seq)

        deconv4 = F.selu(self.deconv4(conv4_lstm))
        deconv4 = self._resize_like(deconv4, conv3_lstm)
        deconv4 = torch.cat([deconv4, conv3_lstm], dim=1)

        deconv3 = F.selu(self.deconv3(deconv4))
        deconv3 = self._resize_like(deconv3, conv2_lstm)
        deconv3 = torch.cat([deconv3, conv2_lstm], dim=1)

        deconv2 = F.selu(self.deconv2(deconv3))
        deconv2 = self._resize_like(deconv2, conv1_lstm)
        deconv2 = torch.cat([deconv2, conv1_lstm], dim=1)

        deconv1 = self.deconv1(deconv2)
        deconv1 = self._resize_like(deconv1, x[:, -1])
        return F.selu(deconv1)

    @staticmethod
    def reconstruction_scores(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (recon - target) ** 2
        return err.flatten(start_dim=1).mean(dim=1)


__all__ = ["ConvLSTMCell", "MSCREDNetwork"]
