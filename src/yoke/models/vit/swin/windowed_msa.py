"""Windowed Multi-headed Self-Attention layers module.

This module defines:
    - WindowMSA
    - ShiftedWindowMSA
    - WindowCosMSA
    - ShiftedWindowCosMSA

Optional Attention Sink (append-and-trim mechanism):
    Instead of adding a bias directly to logits, a learnable sink logit tensor
    (same shape as one key-position slice) is APPENDED as phantom key logits:
        1. Concatenate sink logits onto the real logits along the key dimension.
        2. Apply softmax over the extended dimension (real + sink).
        3. Immediately discard (trim) the sink probabilities so they do not
           contribute to the value aggregation.
    Effect: siphons probability mass from real tokens via softmax denominator
    without altering downstream tensor shapes or requiring K/V cache entries.

"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import numpy as np

from yoke.models.vit.embedding_encoders import RelativePositionEmbed

def _build_attn_sink(
    num_heads: int, window_size: (int, int), device: torch.device | None = None
) -> nn.Parameter:
    """Utility to build the attention sink parameter.

    Shape: (1, num_heads, 1, 1, 1, wh*ww)
        Broadcast over (B, Hw, Ww, query_positions); unique per head & key position.
    """
    wh, ww = window_size
    return nn.Parameter(
        torch.zeros(1, num_heads, 1, 1, 1, wh * ww, device=device),
        requires_grad=True
    )

class WindowMSA(nn.Module):
    """Original Windowed-MSA (non-shifted).

    Applies multi-headed self-attention within non-overlapping windows.

    Attention Sink (optional):
        If use_attention_sink=True, a learnable sink tensor (attn_sink) of shape
        (1, num_heads, 1, 1, 1, wh*ww) is appended as phantom key logits,
        softmax is computed over the concatenated logits, then the sink
        probabilities are trimmed before value aggregation (append-and-trim).
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        super().__init__()
        try:
            msg = "Embedding size not divisible by number of heads!!!"
            assert emb_size % num_heads == 0, msg
        except AssertionError as e:
            e.args += ("Embedding size:", emb_size, "Number of heads:", num_heads)
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[0] % window_size[0] == 0, msg
        except AssertionError as e:
            e.args += (
                "Patch-grid 1:",
                patch_grid_size[0],
                "Window-size 1:",
                window_size[0],
            )
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[1] % window_size[1] == 0, msg
        except AssertionError as e:
            e.args += (
                "Patch-grid 2:",
                patch_grid_size[1],
                "Window-size 2:",
                window_size[1],
            )
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        x = self.linear1(x)
        x = rearrange(
            x, "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        h_dim = self.emb_size / self.num_heads
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)
        wei = self.rel_pos_embed(wei)

        if self.use_attention_sink:
            sink = self.attn_sink
            sink_exp = sink.expand(wei.shape[:-1] + (wei.shape[-1],))
            wei_ext = torch.cat([wei, sink_exp], dim=-1)           # (..., L, 2L)
            attn = F.softmax(wei_ext, dim=-1)[..., :wei.shape[-1]] # trim sink
        else:
            attn = F.softmax(wei, dim=-1)

        wei = attn @ V

        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        return self.linear2(x)

class ShiftedWindowMSA(nn.Module):
    """Shifted Windowed Multi-headed Self-Attention.

    Adds a half-window spatial shift before partitioning to enable cross-window
    interactions (Swin-style). Optional append-and-trim attention sink.
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        super().__init__()
        try:
            assert emb_size % num_heads == 0, "Embedding size not divisible by number of heads!!!"
        except AssertionError as e:
            e.args += ("Embedding size:", emb_size, "Number of heads:", num_heads)
            raise
        try:
            assert patch_grid_size[0] % window_size[0] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 1:", patch_grid_size[0], "Window-size 1:", window_size[0])
            raise
        try:
            assert patch_grid_size[1] % window_size[1] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 2:", patch_grid_size[1], "Window-size 2:", window_size[1])
            raise
        try:
            assert window_size[0] % 2 == 0, "Window height not divisble by 2!!!"
        except AssertionError as e:
            e.args += ("Window height:", window_size[0])
            raise
        try:
            assert window_size[1] % 2 == 0, "Window width not divisble by 2!!!"
        except AssertionError as e:
            e.args += ("Window width:", window_size[1])
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        x = self.linear1(x)
        x = rearrange(
            x, "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )

        x = torch.roll(
            x, (-self.window_size[0] // 2, -self.window_size[1] // 2), dims=(1, 2)
        )

        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        h_dim = self.emb_size / self.num_heads
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)
        wei = self.rel_pos_embed(wei)

        # Shift masking
        row_mask = torch.zeros(
            (self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1])
        ).to(x.device)
        halfIDX = self.window_size[0] * (self.window_size[1] // 2)
        row_mask[-halfIDX:, 0:-halfIDX] = float("-inf")
        row_mask[0:-halfIDX, -halfIDX:] = float("-inf")
        column_mask = rearrange(
            row_mask,
            "(r wh) (c ww) -> (wh r) (ww c)",
            wh=self.window_size[0],
            ww=self.window_size[1],
        )
        wei[:, :, -1, :] += row_mask
        wei[:, :, :, -1] += column_mask

        if self.use_attention_sink:
            sink = self.attn_sink
            sink_exp = sink.expand(wei.shape[:-1] + (wei.shape[-1],))
            wei_ext = torch.cat([wei, sink_exp], dim=-1)
            attn = F.softmax(wei_ext, dim=-1)[..., :wei.shape[-1]]
        else:
            attn = F.softmax(wei, dim=-1)

        wei = attn @ V

        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        return self.linear2(x)

class WindowCosMSA(nn.Module):
    """Cosine-similarity Windowed-MSA with optional append-and-trim sink."""

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        super().__init__()
        try:
            assert emb_size % num_heads == 0, "Embedding size not divisible by number of heads!!!"
        except AssertionError as e:
            e.args += ("Embedding size:", emb_size, "Number of heads:", num_heads)
            raise
        try:
            assert patch_grid_size[0] % window_size[0] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 1:", patch_grid_size[0], "Window-size 1:", window_size[0])
            raise
        try:
            assert patch_grid_size[1] % window_size[1] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 2:", patch_grid_size[1], "Window-size 2:", window_size[1])
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((1, num_heads, 1, 1, 1, 1))), requires_grad=True
        )
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        x = self.linear1(x)
        x = rearrange(
            x, "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        wei = F.normalize(Q, dim=-1) @ F.normalize(K, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01).to(x.device))
        ).exp()
        wei = wei * logit_scale
        wei = self.rel_pos_embed(wei)

        if self.use_attention_sink:
            sink = self.attn_sink
            sink_exp = sink.expand(wei.shape[:-1] + (wei.shape[-1],))
            wei_ext = torch.cat([wei, sink_exp], dim=-1)
            attn = F.softmax(wei_ext, dim=-1)[..., :wei.shape[-1]]
        else:
            attn = F.softmax(wei, dim=-1)

        wei = attn @ V

        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        return self.linear2(x)

class ShiftedWindowCosMSA(nn.Module):
    """Shifted cosine-similarity Windowed-MSA with optional append-and-trim sink."""

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        super().__init__()
        try:
            assert emb_size % num_heads == 0, "Embedding size not divisible by number of heads!!!"
        except AssertionError as e:
            e.args += ("Embedding size:", emb_size, "Number of heads:", num_heads)
            raise
        try:
            assert patch_grid_size[0] % window_size[0] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 1:", patch_grid_size[0], "Window-size 1:", window_size[0])
            raise
        try:
            assert patch_grid_size[1] % window_size[1] == 0, "Patch-grid not divisible by window-size!!!"
        except AssertionError as e:
            e.args += ("Patch-grid 2:", patch_grid_size[1], "Window-size 2:", window_size[1])
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((1, num_heads, 1, 1, 1, 1))), requires_grad=True
        )
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        x = self.linear1(x)
        x = rearrange(
            x, "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )
        x = torch.roll(
            x, (-self.window_size[0] // 2, -self.window_size[1] // 2), dims=(1, 2)
        )
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        wei = F.normalize(Q, dim=-1) @ F.normalize(K, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01).to(x.device))
        ).exp()
        wei = wei * logit_scale
        wei = self.rel_pos_embed(wei)

        # Shift masking
        row_mask = torch.zeros(
            (self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1])
        ).to(x.device)
        halfIDX = self.window_size[0] * (self.window_size[1] // 2)
        row_mask[-halfIDX:, 0:-halfIDX] = float("-inf")
        row_mask[0:-halfIDX, -halfIDX:] = float("-inf")
        column_mask = rearrange(
            row_mask,
            "(r wh) (c ww) -> (wh r) (ww c)",
            wh=self.window_size[0],
            ww=self.window_size[1],
        )
        wei[:, :, -1, :] += row_mask
        wei[:, :, :, -1] += column_mask

        if self.use_attention_sink:
            sink = self.attn_sink
            sink_exp = sink.expand(wei.shape[:-1] + (wei.shape[-1],))
            wei_ext = torch.cat([wei, sink_exp], dim=-1)
            attn = F.softmax(wei_ext, dim=-1)[..., :wei.shape[-1]]
        else:
            attn = F.softmax(wei, dim=-1)

        wei = attn @ V

        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        return self.linear2(x)


if __name__ == "__main__":
    # Simple smoke test
    x = torch.rand(2, 56 * 40, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)

    cfg = dict(
        emb_size=64,
        num_heads=8,
        patch_grid_size=(56, 40),
        window_size=(8, 10),
        use_attention_sink=True,
    )

    print("Input:", x.shape)
    print("WindowMSA:", WindowMSA(**cfg).to(device)(x).shape)
    print("ShiftedWindowMSA:", ShiftedWindowMSA(**cfg).to(device)(x).shape)
    print("WindowCosMSA:", WindowCosMSA(**cfg).to(device)(x).shape)
    print("ShiftedWindowCosMSA:", ShiftedWindowCosMSA(**cfg).to(device)(x).shape)