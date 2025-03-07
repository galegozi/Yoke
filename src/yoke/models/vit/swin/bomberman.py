"""Module for BomberMan network structures.

nn.Module allowing processing of variable channel image input through a SWIN-V2
U-Net architecture then re-embedded as a variable channel image.

This network architecture will serve as the foundation for a hydro-code
emulator.

"""

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from lightning.pytorch import LightningModule

from yoke.models.vit.swin.unet import SwinUnetBackbone
from yoke.models.vit.patch_embed import ParallelVarPatchEmbed
from yoke.models.vit.patch_manipulation import Unpatchify

from yoke.models.vit.aggregate_variables import AggVars
from yoke.models.vit.embedding_encoders import (
    VarEmbed,
    PosEmbed,
    TimeEmbed,
)

from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.torch_training_utils import save_model_and_optimizer_hdf5
from yoke.torch_training_utils import load_model_and_optimizer_hdf5


class LodeRunner(nn.Module):
    """LodeRunner neural network.

    Parallel-patch embedding with SWIN U-Net backbone and
    unpatchification. This module will take in a variable-channel image format
    and output an equivalent variable-channel image formate. This will serves
    as a prototype foundational architecture for multi-material, multi-physics,
    surrogate models.

    Args:
        default_vars (list[str]): List of default variables to be used for training
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        patch_size (tuple[int, int]): Height and width pixel dimensions of patch in
                                      initial embedding.
        emb_dim (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/expand.
        num_heads (int): Number of heads in the MSA layers.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        default_vars: list[str],
        image_size: (int, int) = (1120, 800),
        patch_size: (int, int) = (10, 10),
        embed_dim: int = 128,
        emb_factor: int = 2,
        num_heads: int = 8,
        block_structure: (int, int, int, int) = (1, 1, 3, 1),
        window_sizes: [(int, int), (int, int), (int, int), (int, int)] = [
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales: [(int, int), (int, int), (int, int)] = [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        verbose: bool = False,
    ) -> None:
        """Initialization for class."""
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.emb_factor = emb_factor
        self.num_heads = num_heads
        self.block_structure = block_structure
        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales

        # First embed the image as a sequence of tokenized patches. Each
        # channel is embedded independently.
        self.parallel_embed = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Encode tokens corresponding to each variable with a learnable tag
        self.var_embed_layer = VarEmbed(self.default_vars, self.embed_dim)

        # Aggregate variable tokenizations using an attention mechanism
        self.agg_vars = AggVars(self.embed_dim, self.num_heads)

        # Encode each patch with position information. Position encoding is
        # only index-aware and does not take into account actual spatial
        # information.
        self.pos_embed = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed.num_patches,
        )

        # Encode temporal-offset information using a linear mapping.
        self.temporal_encoding = TimeEmbed(self.embed_dim)

        # Pass encoded patch tokens through a SWIN-Unet structure
        self.unet = SwinUnetBackbone(
            emb_size=self.embed_dim,
            emb_factor=self.emb_factor,
            patch_grid_size=self.parallel_embed.grid_size,
            block_structure=self.block_structure,
            num_heads=self.num_heads,
            window_sizes=self.window_sizes,
            patch_merge_scales=self.patch_merge_scales,
            verbose=verbose,
        )

        # Linear embed the last dimension into V*p_h*p_w
        self.linear4unpatch = nn.Linear(
            self.embed_dim, self.max_vars * self.patch_size[0] * self.patch_size[1]
        )

        # Unmap the tokenized embeddings to variables and images.
        self.unpatch = Unpatchify(
            total_num_vars=self.max_vars,
            patch_grid_size=self.parallel_embed.grid_size,
            patch_size=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for LodeRunner."""
        # WARNING!: Most likely the `in_vars` and `out_vars` need to be tensors
        # of integers corresponding to variables in the `default_vars` list.

        # First embed input
        # varIDXs = self.var_embed_layer.get_var_ids(tuple(in_vars), x.device)
        x = self.parallel_embed(x, in_vars)

        # Encode variables
        x = self.var_embed_layer(x, in_vars)

        # Aggregate variables
        x = self.agg_vars(x)

        # Encode patch positions, spatial information
        x = self.pos_embed(x)

        # Encode temporal information
        x = self.temporal_encoding(x, lead_times)

        # Pass through SWIN-V2 U-Net encoder
        x = self.unet(x)

        # Use linear map to remap to correct variable and patchsize dimension
        x = self.linear4unpatch(x)

        # Unpatchify back to original shape
        x = self.unpatch(x)

        # Select only entries corresponding to out_vars for loss
        # out_var_ids = self.var_embed_layer.get_var_ids(tuple(out_vars), x.device)
        preds = x[:, out_vars]

        return preds


class Lightning_LodeRunner(LightningModule):
    """Lightning wrapper for LodeRunner.

    Wrap LodeRunner torch.nn.Module class in a lightning.LightningModule for
    ease of parallelization and encapsulation of training strategy.

    Args:
        model (nn.Module): Pre-initialized nn.Module to wrap
        in_vars (torch.Tensor): Input channels to train LodeRunner on
        out_vars (torch.Tensor): Output channels to train LodeRunner on
        learning_rate (float): Initial learning rate for optimizer. Ignored if a
                               scheduler is used.
        lrscheduler (_LRScheduler): Learning-rate scheduler to use with optimizer
        scheduler_params (dict): Keyword arguments to initialize scheduler

    """

    def __init__(
        self,
        model: nn.Module,
        in_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        out_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        learning_rate: float = 1e-3,
        lrscheduler: _LRScheduler = None,
        scheduler_params: dict = None,
    ) -> None:
        """Initialization for Lightning wrapper."""
        super().__init__()
        self.model = model
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.learning_rate = learning_rate
        self.lrscheduler = lrscheduler
        self.scheduler_params = scheduler_params or {}
        self.loss_fn = nn.MSELoss(reduction="none")

        # These attributes will be dynamically set during training
        self.load_h5_chkpt = "./dummy_load.hdf5"
        self.save_h5_chkpt = "./dummy_save.hdf5"

    def forward(self, X: torch.Tensor, lead_times: torch.Tensor) -> torch.Tensor:
        """Forward method for Lightning wrapper."""
        # Forward pass through the custom model
        return self.model(X, self.in_vars, self.out_vars, lead_times)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        # Assume batch includes all required inputs
        start_img, end_img, lead_times = batch  # Unpack batch
        preds = self(start_img, lead_times)  # Forward pass

        # Per-sample MSE
        losses = self.loss_fn(preds, end_img)
        if hasattr(self, "Trainer"):  # Only log if there is a Trainer
            self.log("train_loss_per_sample", losses, on_epoch=True, on_step=True)

        batch_loss = losses.mean()
        if hasattr(self, "Trainer"):  # Only log if there is a Trainer
            self.log("train_loss", batch_loss)

        return batch_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Execute validation step."""
        start_img, end_img, lead_times = batch  # Unpack batch
        preds = self(start_img, lead_times)  # Forward pass

        # Per-sample MSE
        losses = self.loss_fn(preds, end_img)
        if hasattr(self, "Trainer"):  # Only log if there is a Trainer
            self.log("val_loss_per_sample", losses, on_epoch=True, on_step=True)

        batch_loss = losses.mean()
        if hasattr(self, "Trainer"):  # Only log if there is a Trainer
            self.log("val_loss", batch_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Setup optimizer with scheduler."""
        # Optimizer setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        if self.lrscheduler:
            # Initialize LR scheduler
            scheduler = self.lrscheduler(optimizer, **self.scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Step scheduler every batch.
                    "frequency": 1,  # Step every batch (default for "step")
                },
            }

        return optimizer

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Custom save checkpoint."""
        epoch = checkpoint.get("epoch", 0)  # Returns 0 if 'epoch' key doesn't exist.
        filepath = self.save_h5_chkpt
        save_model_and_optimizer_hdf5(
            self.model,
            self.optimizers()[0],
            epoch=epoch,
            filepath=filepath,
        )
        self.print(f"Saved HDF5 checkpoint: {filepath}")

    def on_load_checkpoint(self) -> None:
        """Custom load checkpoint."""
        filepath = self.load_h5_chkpt
        try:
            loaded_epoch = load_model_and_optimizer_hdf5(
                self.model,
                self.optimizers()[0],
                filepath=filepath,
            )
            self.current_epoch_override = loaded_epoch
            self.print(f"Loaded HDF5 checkpoint: {filepath}")
        except FileNotFoundError:
            self.print(f"Checkpoint file not found: {filepath}. Starting fresh!")


if __name__ == "__main__":
    from yoke.torch_training_utils import count_torch_params

    device = "cuda" if torch.cuda.is_available() else "cpu"

    default_vars = [
        "cu_pressure",
        "cu_density",
        "cu_temperature",
        "al_pressure",
        "al_density",
        "al_temperature",
        "ss_pressure",
        "ss_density",
        "ss_temperature",
        "ply_pressure",
        "ply_density",
        "ply_temperature",
        "air_pressure",
        "air_density",
        "air_temperature",
        "hmx_pressure",
        "hmx_density",
        "hmx_temperature",
        "r_vel",
        "z_vel",
    ]

    # (B, C, H, W)
    x = torch.rand(5, 4, 1120, 800)
    x = x.type(torch.FloatTensor).to(device)

    lead_times = torch.rand(5).to(device)  # Lead time for each entry in batch
    # x_vars = ["cu_density", "ss_density", "ply_density", "air_density"]
    x_vars = torch.tensor([1, 7, 10, 13]).to(device)

    # out_vars = ["cu_density", "ss_density", "ply_density", "air_density"]
    out_vars = torch.tensor([1, 7, 10, 13]).to(device)

    # Common model setup for LodeRunner
    #
    # NOTE: For half-image `image_size = (1120, 400)` can just halve the second
    # patch_size dimension.
    emb_factor = 2
    patch_size = (10, 10)
    image_size = (1120, 800)
    num_heads = 8
    window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)]
    patch_merge_scales = [(2, 2), (2, 2), (2, 2)]

    # Tiny size
    embed_dim = 96
    block_structure = (1, 1, 3, 1)

    # Test LodeRunner architecture.
    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    loderunner_out = lode_runner(x, x_vars, out_vars, lead_times)
    print("LodeRunner-tiny output shape:", loderunner_out.shape)
    print("LodeRunner-tiny output has NaNs:", torch.isnan(loderunner_out).any())
    print("LodeRunner-tiny parameters:", count_torch_params(lode_runner, trainable=True))

    # Test lightning wrapper initialization.
    L_loderunner = Lightning_LodeRunner(
        lode_runner,
        in_vars=x_vars,
        out_vars=out_vars,
        lrscheduler=CosineWithWarmupScheduler,
        scheduler_params={
            "warmup_steps": 500,
            "anchor_lr": 1e-3,
            "terminal_steps": 1000,
            "num_cycles": 0.5,
            "min_fraction": 0.5,
            "last_epoch": 0,
        },
    )
    L_loderunner_out = L_loderunner(x, lead_times)
    print("Lightning LodeRunner-tiny output shape:", L_loderunner_out.shape)

    # Small size
    embed_dim = 96
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-small parameters:", count_torch_params(lode_runner, trainable=True)
    )

    # Big size
    embed_dim = 128
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print("LodeRunner-big parameters:", count_torch_params(lode_runner, trainable=True))

    # Large size
    embed_dim = 192
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-large parameters:", count_torch_params(lode_runner, trainable=True)
    )

    # Huge size
    embed_dim = 352
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print("LodeRunner-huge parameters:", count_torch_params(lode_runner, trainable=True))

    # Giant size
    embed_dim = 512
    block_structure = (1, 1, 11, 2)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-giant parameters:", count_torch_params(lode_runner, trainable=True)
    )
