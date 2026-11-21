"""Encoder + swappable heads.

Shapes:  x [B, C, T]  ->  tokens z [B, C, P, H]   (P = T / patch_size patches)

* The conv stem is applied to every channel independently with shared weights
  (a 1-channel conv), so the encoder works for any number of channels C and
  any length T.
* The transformer alternates attention over time (within each channel) and
  attention over channels (within each patch), so it mixes information
  across both axes without assuming a fixed C or P.
* Heads are deliberately light, so that the encoder has to do most of the work:
    - ReconstructionHead: one linear layer, token -> raw patch values
    - ClassificationHead: masked mean pooling + one linear layer
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ConvPatchEmbed(nn.Module):
    """Per-channel conv stem: [B, C, T] (+ mask indicator) -> [B, C, P, H]."""

    def __init__(self, hidden: int, patch_size: int, stem_kernel: int = 7):
        super().__init__()
        self.patch_size = patch_size
        # input channel 0 = (masked) signal, channel 1 = "this value is masked" indicator
        self.stem = nn.Sequential(
            nn.Conv1d(2, hidden // 2, stem_kernel, padding=stem_kernel // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, x, mask_indicator):
        B, C, T = x.shape
        inp = torch.stack([x, mask_indicator], dim=2).reshape(B * C, 2, T)
        z = self.stem(inp)                                  # [B*C, H, P]
        return z.transpose(1, 2).reshape(B, C, -1, z.shape[1])


def sinusoidal_positions(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class AxialBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        layer = lambda: nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.time_attn, self.channel_attn = layer(), layer()

    def forward(self, z, patch_pad, channel_pad):
        B, C, P, H = z.shape
        z = self.time_attn(z.reshape(B * C, P, H),
                           src_key_padding_mask=patch_pad.repeat_interleave(C, 0)).reshape(B, C, P, H)
        z = z.transpose(1, 2).reshape(B * P, C, H)
        z = self.channel_attn(z, src_key_padding_mask=channel_pad.repeat_interleave(P, 0))
        return z.reshape(B, P, C, H).transpose(1, 2)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3, heads: int = 4, patch_size: int = 32,
                 dropout: float = 0.1, n_channel_ids: int = 0):
        """`n_channel_ids` > 0 adds a learned embedding per variable (by index into a
        channel-name vocabulary) so the model can tell physical variables apart.
        Set to 0 for a fully channel-agnostic model."""
        super().__init__()
        self.patch_size, self.hidden = patch_size, hidden
        self.embed = ConvPatchEmbed(hidden, patch_size)
        self.channel_emb = nn.Embedding(n_channel_ids, hidden) if n_channel_ids > 0 else None
        self.blocks = nn.ModuleList([AxialBlock(hidden, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)

    def patch_mask(self, time_mask):
        """[B, T] valid-timestep mask -> [B, P] valid-patch mask (patch valid if any step is real)."""
        B, T = time_mask.shape
        return time_mask.reshape(B, T // self.patch_size, self.patch_size).any(-1)

    def forward(self, x, time_mask, channel_mask, value_mask=None, channel_ids=None):
        """value_mask: [B, C, T] bool, True where values are hidden from the encoder."""
        if value_mask is None:
            value_mask = torch.zeros_like(x, dtype=torch.bool)
        x = x.masked_fill(value_mask, 0.0)
        z = self.embed(x, value_mask.float())                         # [B, C, P, H]
        B, C, P, H = z.shape
        z = z + sinusoidal_positions(P, H, z.device)[None, None]
        if self.channel_emb is not None and channel_ids is not None:
            z = z + self.channel_emb(channel_ids)[:, :, None, :]
        patch_valid = self.patch_mask(time_mask)
        for blk in self.blocks:
            z = blk(z, ~patch_valid, ~channel_mask)
        return self.norm(z), patch_valid


class ReconstructionHead(nn.Module):
    """Lightweight decoder: token -> the patch_size raw values it covers."""

    def __init__(self, hidden: int, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden, patch_size)

    def forward(self, z, patch_valid=None, channel_mask=None):
        B, C, P, _ = z.shape
        return self.proj(z).reshape(B, C, P * self.proj.out_features)   # [B, C, T]


class ClassificationHead(nn.Module):
    """Masked mean pooling over (channels, patches) + linear classifier."""

    def __init__(self, hidden: int, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, z, patch_valid, channel_mask):
        w = (channel_mask[:, :, None] & patch_valid[:, None, :]).float()[..., None]  # [B, C, P, 1]
        pooled = (z * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1.0)
        return self.head(pooled)
