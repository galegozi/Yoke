"""Windowed Multi-headed Self-Attention layers module.

This module defines:
    - WindowMSA
    - ShiftedWindowMSA
    - WindowCosMSA
    - ShiftedWindowCosMSA

A modification has been made to optionally include an *attention sink*:
a learnable per-head, per-key-position bias added to the attention logits
(before softmax) inside each window. This allows the model to learn a
persistent prior over window key positions (akin to adding a trainable
parameter to softmax logits as seen in some GPT-style OSS implementations).

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
    """Original Windowed-MSA.

    Applies multi-headed self-attention within non-overlapping windows.

    Attention Sink (optional):
        If use_attention_sink=True, a learnable bias (attn_sink) of shape
        (1, num_heads, 1, 1, 1, wh*ww) is added to the attention logits
        after relative position embedding but before softmax. This provides a
        per-head, per-key positional prior inside each window.

    Args:
        emb_size (int): Token embedding dimension.
        num_heads (int): Number of attention heads.
        patch_grid_size (int, int): (H, W) of token grid.
        window_size (int, int): (wh, ww) window size dividing the grid evenly.
        use_attention_sink (bool): Enable learnable attention sink bias.
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        """Initialization method."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Embedding size not divisible by number of heads!!!"
            assert emb_size % num_heads == 0, msg
        except AssertionError as e:
            msg_tuple = ("Embedding size:", emb_size, "Number of heads:", num_heads)
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[0] % window_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 1:",
                patch_grid_size[0],
                "Window-size 1:",
                window_size[0],
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[1] % window_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 2:",
                patch_grid_size[1],
                "Window-size 2:",
                window_size[1],
            )
            e.args += msg_tuple
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        # QKV embedding
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)

        # Linear output embedding.
        self.linear2 = nn.Linear(emb_size, emb_size)

        # Initialize relative position embedding
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Window-MSA."""
        # B: Batch-size
        # L: Number of tokens = H*W
        # C: token length or embedding dimension, i.e. emb_size
        _, L, _ = x.shape

        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        # Map C to the Q,K,V matrices with linear embedding
        x = self.linear1(x)

        # Resulting tensor has size (B, L, 3*C). This tensor is broken into 2D
        # arrays of the tokens (Q, K, V) embeddings.  Each embedding is
        # rearranged into a 2D spatial structure.
        x = rearrange(
            x,
            "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )

        # The height and width dimensions are now *windowed*. There are now
        # Hw*Ww windows, each of size wh*ww. The windowed tokens are arranged
        # in a 2D, Hw x Hw, grid. The embedding dimension is separated into
        # heads and the head dimension is moved to the dimension right after
        # the batch-size.
        #
        # NOTE: The window size must divide the height and width evenly.
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)  # Corresponds to k in the rearrange above.
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # wei now represents the attention weights between every pair of
        # patches/tokens within each of the Hw x Ww windows. There are wh*ww
        # patches/tokens within each window so (wh*ww, wh*ww) pairs so
        # shape(wei) = (B, H, Wh, Ww, wh*ww, wh*ww)
        h_dim = self.emb_size / self.num_heads
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)

        # Relative position embedding.
        wei = self.rel_pos_embed(wei)

        if self.use_attention_sink:
            wei = wei + self.attn_sink

        # Passing dim=-1 to softmax ensures that the softmax operation is
        # applied along the last dimension of the wei tensor, which corresponds
        # to the attention weights between tokens within the same window.
        wei = F.softmax(wei, dim=-1) @ V

        # Recombine heads and windows
        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        # Recombine 2D token grid.
        x = rearrange(x, "b h w c -> b (h w) c")

        # Pass through a linear embedding.
        return self.linear2(x)


class ShiftedWindowMSA(nn.Module):
    """Shifted Windowed Multi-headed Self-Attention.

    Adds a half-window spatial shift before window partitioning to permit
    cross-window interactions.

    Attention Sink (optional):
        Same behavior as in WindowMSA (see above).

    Args:
        emb_size (int)
        num_heads (int)
        patch_grid_size (int, int)
        window_size (int, int): Each dim must be even.
        use_attention_sink (bool)
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        """Initialization method."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Embedding size not divisible by number of heads!!!"
            assert emb_size % num_heads == 0, msg
        except AssertionError as e:
            msg_tuple = ("Embedding size:", emb_size, "Number of heads:", num_heads)
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[0] % window_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 1:",
                patch_grid_size[0],
                "Window-size 1:",
                window_size[0],
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[1] % window_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 2:",
                patch_grid_size[1],
                "Window-size 2:",
                window_size[1],
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Window height not divisble by 2!!!"
            assert window_size[0] % 2 == 0, msg
        except AssertionError as e:
            msg_tuple = ("Window height:", window_size[0])
            e.args += msg_tuple
            raise

        try:
            msg = "Window width not divisble by 2!!!"
            assert window_size[1] % 2 == 0, msg
        except AssertionError as e:
            msg_tuple = ("Window width:", window_size[1])
            e.args += msg_tuple
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        # QKV embedding
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)

        # Linear output embedding.
        self.linear2 = nn.Linear(emb_size, emb_size)

        # Initialize relative position embedding
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Windowed MSA."""
        # B: Batch-size
        # L: Number of tokens = H*W
        # C: token length or embedding dimension, i.e. emb_size
        _, L, _ = x.shape

        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        # Map C to the Q,K,V matrices with linear embedding
        x = self.linear1(x)

        # Resulting tensor has size (B, L, 3*C). This tensor is broken into 2D
        # arrays of the tokens (Q, K, V) embeddings. Each embedding is
        # rearranged into a 2D spatial structure.
        x = rearrange(
            x,
            "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )

        # Roll the QKV embedding entries along the 2D-spatial dimensions by
        # half the window size. Elements are shifted along the *dims*
        # dimensions, those elements shifted beyond the last position are
        # re-introduced at the first position.
        x = torch.roll(
            x, (-self.window_size[0] // 2, -self.window_size[1] // 2), dims=(1, 2)
        )

        # The height and width dimensions are now *windowed*. There are now
        # Hw*Ww windows, each of size wh*ww. The windowed tokens are arranged
        # in a 2D, Hw x Hw, grid. The embedding dimension is separated into
        # heads and the head dimension is moved to the dimension right after
        # the batch-size.
        #
        # NOTE: The window size must divide the height and width evenly.
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)  # Corresponds to k in the rearrange above.
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # wei now represents the attention weights between every pair of
        # patches/tokens within each of the Hw x Ww windows. There are wh*ww
        # patches/tokens within each window so (wh*ww, wh*ww) pairs so
        # shape(wei) = (B, H, Wh, Ww, wh*ww, wh*ww)
        h_dim = self.emb_size / self.num_heads
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)

        # Relative position embedding.
        wei = self.rel_pos_embed(wei)

        # Masking to ensure that tokens on opposite edges of the window prior
        # to the shift do not interact.
        #
        # NOTE: The size of the attention weights is (B, H, Wh, Ww, wh*ww,
        # wh*ww) and row_mask and column_mask are size (wh*ww, wh*ww).
        row_mask = torch.zeros(
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
        ).to(x.device)  # .cuda() only if cuda enabled

        # Set bottom left quarter of mask to *-inf*
        halfIDX = self.window_size[0] * (self.window_size[1] // 2)
        row_mask[-halfIDX:, 0:-halfIDX] = float("-inf")
        # Set top right quarter of mask to *-inf*
        row_mask[0:-halfIDX, -halfIDX:] = float("-inf")

        # Rearranging creates wh*ww x wh*ww matrix made up of wh x ww
        # sub-matrices each having thier bottom left and top right quadrants
        # set to *-inf*.
        column_mask = rearrange(
            row_mask,
            "(r wh) (c ww) -> (wh r) (ww c)",
            wh=self.window_size[0],
            ww=self.window_size[1],
        )

        ############################################
        # Uncomment to observe structure of masks...
        # print('row mask:')
        # for i in range(row_mask.shape[0]):
        #     print(row_mask[i, :].tolist())

        # print('column mask:')
        # for i in range(column_mask.shape[0]):
        #     print(column_mask[i, :].tolist())
        ############################################

        # Entries having value *-inf* are zeroed after passing through softmax.
        #
        # For every batch entry and head add row_mask to all entries
        # corresponding to the last Wh-dimension entry.
        wei[:, :, -1, :] += row_mask

        # For every batch entry and head add col_mask to all entries
        # corresponding to the last Ww-dimension entry.
        wei[:, :, :, -1] += column_mask

        if self.use_attention_sink:
            wei = wei + self.attn_sink

        # Passing dim=-1 to softmax ensures that the softmax operation is
        # applied along the last dimension of the wei tensor, which corresponds
        # to the attention weights between tokens within the same window.
        wei = F.softmax(wei, dim=-1) @ V

        # Recombine heads and windows
        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        # Recombine 2D token grid.
        x = rearrange(x, "b h w c -> b (h w) c")

        # Pass through a linear embedding.
        return self.linear2(x)


class WindowCosMSA(nn.Module):
    """Cosine-Attention Windowed-MSA.

    Uses cosine similarity with a per-head learnable logit scale.

    Attention Sink:
        Optional additive bias (see WindowMSA) applied after relative position
        embedding and before softmax.

    Args:
        emb_size (int)
        num_heads (int)
        patch_grid_size (int, int)
        window_size (int, int)
        use_attention_sink (bool)
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        """Initialization method for Cos-WindowMSA."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Embedding size not divisible by number of heads!!!"
            assert emb_size % num_heads == 0, msg
        except AssertionError as e:
            msg_tuple = ("Embedding size:", emb_size, "Number of heads:", num_heads)
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[0] % window_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 1:",
                patch_grid_size[0],
                "Window-size 1:",
                window_size[0],
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[1] % window_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 2:",
                patch_grid_size[1],
                "Window-size 2:",
                window_size[1],
            )
            e.args += msg_tuple
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        # Learnable per-head attention scaling
        # Multiplies attn.shape=(B, num_heads, Hw, Ww, wh*ww, wh*ww)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((1, num_heads, 1, 1, 1, 1))), requires_grad=True
        )

        # QKV embedding
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)

        # Linear output embedding.
        self.linear2 = nn.Linear(emb_size, emb_size)

        # Initialize relative position embedding
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Cos-WindowMSA."""
        # B: Batch-size
        # L: Number of tokens = H*W
        # C: token length or embedding dimension, i.e. emb_size
        _, L, _ = x.shape

        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        # Map C to the Q,K,V matrices with linear embedding
        x = self.linear1(x)

        # Resulting tensor has size (B, L, 3*C). This tensor is broken into 2D
        # arrays of the tokens (Q, K, V) embeddings.  Each embedding is
        # rearranged into a 2D spatial structure.
        x = rearrange(
            x,
            "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )

        # The height and width dimensions are now *windowed*. There are now
        # Hw*Ww windows, each of size wh*ww. The windowed tokens are arranged
        # in a 2D, Hw x Hw, grid. The embedding dimension is separated into
        # heads and the head dimension is moved to the dimension right after
        # the batch-size.
        #
        # NOTE: The window size must divide the height and width evenly.
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        Q, K, V = x.chunk(3, dim=6)  # Corresponds to k in the rearrange above.
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # wei now represents the attention weights between every pair of
        # patches/tokens within each of the Hw x Ww windows. There are wh*ww
        # patches/tokens within each window so (wh*ww, wh*ww) pairs so
        # shape(wei) = (B, H, Wh, Ww, wh*ww, wh*ww)
        #
        # NOTE: Cosine attention is used with a learnable, per-head
        # scaling. Since cos(theta(ab))= <a,b>/|a|*|b| this amounts to a normalized
        # dot-product attention.
        wei = F.normalize(Q, dim=-1) @ F.normalize(K, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01).to(x.device))
        ).exp()
        wei = wei * logit_scale

        # Relative position embedding.
        wei = self.rel_pos_embed(wei)

        if self.use_attention_sink:
            wei = wei + self.attn_sink

        # Passing dim=-1 to softmax ensures that the softmax operation is
        # applied along the last dimension of the wei tensor, which corresponds
        # to the attention weights between tokens within the same window.
        wei = F.softmax(wei, dim=-1) @ V

        # Recombine heads and windows
        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        # Recombine 2D token grid.
        x = rearrange(x, "b h w c -> b (h w) c")

        # Pass through a linear embedding.
        return self.linear2(x)


class ShiftedWindowCosMSA(nn.Module):
    """Cosine-Attention Shifted Window-MSA.

    Adds spatial shift (like ShiftedWindowMSA) plus cosine attention with
    logit scaling (SWIN-V2 style), plus optional attention sink.

    Args:
        emb_size (int)
        num_heads (int)
        patch_grid_size (int, int)
        window_size (int, int)
        use_attention_sink (bool)
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
        use_attention_sink: bool = True,
    ) -> None:
        """Initialization for Cos-WindowMSA."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Embedding size not divisible by number of heads!!!"
            assert emb_size % num_heads == 0, msg
        except AssertionError as e:
            msg_tuple = ("Embedding size:", emb_size, "Number of heads:", num_heads)
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[0] % window_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 1:",
                patch_grid_size[0],
                "Window-size 1:",
                window_size[0],
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid not divisible by window-size!!!"
            assert patch_grid_size[1] % window_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid 2:",
                patch_grid_size[1],
                "Window-size 2:",
                window_size[1],
            )
            e.args += msg_tuple
            raise

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size
        self.use_attention_sink = use_attention_sink

        # Learnable per-head attention scaling
        # Multiplies attn.shape=(B, num_heads, Hw, Ww, wh*ww, wh*ww)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((1, num_heads, 1, 1, 1, 1))), requires_grad=True
        )

        # QKV embedding
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)

        # Linear output embedding.
        self.linear2 = nn.Linear(emb_size, emb_size)

        # Initialize relative position embedding
        if self.use_attention_sink:
            self.attn_sink = _build_attn_sink(num_heads, window_size)
        else:
            self.register_parameter("attn_sink", None)
        self.rel_pos_embed = RelativePositionEmbed(window_size=self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Cos-WindowMSA."""
        # B: Batch-size
        # L: Number of tokens = H*W
        # C: token length or embedding dimension, i.e. emb_size
        _, L, _ = x.shape

        assert self.patch_grid_size[0] * self.patch_grid_size[1] == L

        # Map C to the Q,K,V matrices with linear embedding
        x = self.linear1(x)

        # Resulting tensor has size (B, L, 3*C). This tensor is broken into 2D
        # arrays of the tokens (Q, K, V) embeddings. Each embedding is
        # rearranged into a 2D spatial structure.
        x = rearrange(
            x,
            "b (h w) (c k) -> b h w c k",
            h=self.patch_grid_size[0],
            w=self.patch_grid_size[1],
            k=3,
            c=self.emb_size,
        )

        # Roll the QKV embedding entries along the 2D-spatial dimensions by
        # half the window size. Elements are shifted along the *dims*
        # dimensions, those elements shifted beyond the last position are
        # re-introduced at the first position.
        x = torch.roll(
            x, (-self.window_size[0] // 2, -self.window_size[1] // 2), dims=(1, 2)
        )

        # The height and width dimensions are now *windowed*. There are now
        # Hw*Ww windows, each of size wh*ww. The windowed tokens are arranged
        # in a 2D, Hw x Hw, grid. The embedding dimension is separated into
        # heads and the head dimension is moved to the dimension right after
        # the batch-size.
        #
        # NOTE: The window size must divide the height and width evenly.
        x = rearrange(
            x,
            "b (Hw wh) (Ww ww) (e H) k -> b H Hw Ww (wh ww) e k",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )
        Q, K, V = x.chunk(3, dim=6)  # Corresponds to k in the rearrange above.
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # wei now represents the attention weights between every pair of
        # patches/tokens within each of the Hw x Ww windows. There are wh*ww
        # patches/tokens within each window so (wh*ww, wh*ww) pairs so
        # shape(wei) = (B, H, Wh, Ww, wh*ww, wh*ww)
        #
        # NOTE: Cosine attention is used with a learnable, per-head
        # scaling. Since cos(theta(ab))= <a,b>/|a|*|b| this amounts to a normalized
        # dot-product attention.
        wei = F.normalize(Q, dim=-1) @ F.normalize(K, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01).to(x.device))
        ).exp()
        wei = wei * logit_scale

        # Relative position embedding.
        wei = self.rel_pos_embed(wei)

        # Masking to ensure that tokens on opposite edges of the window prior
        # to the shift do not interact.
        #
        # NOTE: The size of the attention weights is (B, H, Wh, Ww, wh*ww,
        # wh*ww) and row_mask and column_mask are size (wh*ww, wh*ww).
        row_mask = torch.zeros(
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
        ).to(x.device)

        # Set bottom left quarter of mask to *-inf*
        halfIDX = self.window_size[0] * (self.window_size[1] // 2)
        row_mask[-halfIDX:, 0:-halfIDX] = float("-inf")
        # Set top right quarter of mask to *-inf*
        row_mask[0:-halfIDX, -halfIDX:] = float("-inf")

        # Rearranging creates wh*ww x wh*ww matrix made up of wh x ww
        # sub-matrices each having thier bottom left and top right quadrants
        # set to *-inf*.
        column_mask = rearrange(
            row_mask,
            "(r wh) (c ww) -> (wh r) (ww c)",
            wh=self.window_size[0],
            ww=self.window_size[1],
        )

        ############################################
        # Uncomment to observe structure of masks...
        # print('row mask:')
        # for i in range(row_mask.shape[0]):
        #     print(row_mask[i, :].tolist())

        # print('column mask:')
        # for i in range(column_mask.shape[0]):
        #     print(column_mask[i, :].tolist())
        ############################################

        # Entries having value *-inf* are zeroed after passing through softmax.
        #
        # For every batch entry and head add row_mask to all entries
        # corresponding to the last Wh-dimension entry.
        wei[:, :, -1, :] += row_mask

        # For every batch entry and head add col_mask to all entries
        # corresponding to the last Ww-dimension entry.
        wei[:, :, :, -1] += column_mask

        if self.use_attention_sink:
            wei = wei + self.attn_sink

        # Passing dim=-1 to softmax ensures that the softmax operation is
        # applied along the last dimension of the wei tensor, which corresponds
        # to the attention weights between tokens within the same window.
        wei = F.softmax(wei, dim=-1) @ V

        # Recombine heads and windows
        x = rearrange(
            wei,
            "b H Hw Ww (wh ww) e -> b (Hw wh) (Ww ww) (H e)",
            wh=self.window_size[0],
            ww=self.window_size[1],
            H=self.num_heads,
        )

        # Recombine 2D token grid.
        x = rearrange(x, "b h w c -> b (h w) c")

        # Pass through a linear embedding.
        return self.linear2(x)


if __name__ == "__main__":
    # Usage Example.
    x = torch.rand(3, 56 * 40, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    num_heads = 8
    emb_size = 64
    window_size = (8, 10)
    patch_grid_size = (56, 40)

    model_WMSA = WindowMSA(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
        use_attention_sink=True,
    ).to(device)
    model_SWMSA = ShiftedWindowMSA(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
        use_attention_sink=True,
    ).to(device)

    print("Input shape:", x.shape)
    print("WMSA shape:", model_WMSA(x).shape)
    print("SWMSA shape:", model_SWMSA(x).shape)
