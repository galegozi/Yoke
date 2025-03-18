"""Initial training setup for Diffusion Model on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn
import random
import logging
import numpy as np

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr
from yoke.parallel_utils import LodeRunner_DataParallel
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli
import yoke.helpers.logger as ylogger

#############################################
# Inputs
#############################################
descr_str = (
    "Trains Diffusion Model architecture with channel subsampling with single-timstep input"
    " and output of the lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial Diffusion Model Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_step_lr_scheduler_args(parser=parser)
parser = cli.add_ch_subsampling_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

############################################
# Select n channles randomly for an epoch
############################################
def rand_channel_map(
        max_number_channels: int,
        num_subchannels: int,
        seed: int = None,
        _seed_set: list[bool] = [False]
) -> list:
    """Choose list of subsampled channels."""
    if num_subchannels > max_number_channels:
        raise ValueError("Subsampled channels cannot be greater than maximum channels.")

    if seed is not None and not _seed_set[0]:
        random.seed(seed)
        _seed_set[0] = True  # Mark the seed as set

    return sorted(random.sample(range(0, max_number_channels), num_subchannels))

############################################
# Define the model architecture
############################################
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.attention1 = SelfAttention(64)
        self.encoder2 = self.conv_block(64, 128)
        self.attention2 = SelfAttention(128)
        self.encoder3 = self.conv_block(128, 256)
        self.attention3 = SelfAttention(256)
        self.encoder4 = self.conv_block(256, 512)
        self.attention4 = SelfAttention(512)
        self.bottleneck = self.conv_block(512, 1024)
        self.attention5 = SelfAttention(1024)
        self.decoder4 = self.up_conv_block(1024, 512)
        self.attention6 = SelfAttention(512)
        self.decoder3 = self.up_conv_block(512, 256)
        self.attention7 = SelfAttention(256)
        self.decoder2 = self.up_conv_block(256, 128)
        self.attention8 = SelfAttention(128)
        self.decoder1 = self.up_conv_block(128, 64)
        self.attention9 = SelfAttention(64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = self.attention1(enc1)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc2 = self.attention2(enc2)
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc3 = self.attention3(enc3)
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, kernel_size=2))
        enc4 = self.attention4(enc4)
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, kernel_size=2))
        bottleneck = self.attention5(bottleneck)
        dec4 = self.decoder4(bottleneck)
        dec4 = self.attention6(dec4 + bottleneck)
        dec3 = self.decoder3(dec4 + enc4)
        dec3 = self.attention7(dec3 + dec4)
        dec2 = self.decoder2(dec3 + enc3)
        dec2 = self.attention8(dec2 + dec3)
        dec1 = self.decoder1(dec2 + enc2)
        dec1 = self.attention9(dec1 + dec2)
        return self.final_conv(dec1 + enc1)

def add_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    return images + noise

if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    ylogger.configure_logger("yoke_logger", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training Parameters
    anchor_lr = args.anchor_lr
    num_cycles = args.num_cycles
    min_fraction = args.min_fraction
    terminal_steps = args.terminal_steps
    warmup_steps = args.warmup_steps

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor

    # Epoch Parameters
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    train_per_val = args.TRAIN_PER_VAL
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################
    hydro_fields =[
        'density_case',
        'density_cushion',
        'density_maincharge',
        'density_outside_air',
        'density_striker',
        'density_throw',
        'Uvelocity',
        'Wvelocity',
    ]

    model = UNet(
        in_channels=len(hydro_fields),
        out_channels=len(hydro_fields),
    )

    print("UNet parameters:", tr.count_torch_params(model, trainable=True))
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    if args.multigpu:
        model = LodeRunner_DataParallel(model)

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #############################################
    # LR scheduler
    #############################################
    # We will take a scheduler step every back-prop step so the number of steps
    # is the number of previous batches.
    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)
    LRsched = CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=anchor_lr,
        terminal_steps=terminal_steps,
        warmup_steps=warmup_steps,
        num_cycles=num_cycles,
        min_fraction=min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # For reproducibility of channel map per epoch
    SEED = 42

    max_channels = len(hydro_fields)
    ylogger.logger.info(f"Max Channel  : {max_channels}")
    channel_map_size = args.channel_map_size

    # Change hydrofields to array to enable slicing with channel map
    hydro_fields = np.array(hydro_fields)
    for epochIDX in range(starting_epoch, ending_epoch):
        # Randomly select 'channel_map_size' number of channels from for the epoch
        channel_map = rand_channel_map(max_channels, channel_map_size, SEED)

        # Blank spaces in the log strings are for next line alignment
        log_str = (
            f"Epoch {epochIDX:04d}, "
            f"Nchannels:{channel_map_size:03d}/{max_channels}, \n          "
            f"Channel Map:{channel_map}"
            )
        ylogger.logger.info(log_str)

        #############################################
        # Initialize Data
        # For varying channels subset per epoch, the
        # data must be initialized for each epoch.
        #############################################
        train_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=train_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            half_image=True,
            hydro_fields=hydro_fields[channel_map],
        )
        val_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=validation_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            half_image=True,
            hydro_fields=hydro_fields[channel_map],
        )
        test_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=test_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            half_image=True,
            hydro_fields=hydro_fields[channel_map],
        )

        print("Datasets initialized...")

        # Setup Dataloaders
        train_dataloader = tr.make_dataloader(
            train_dataset,
            batch_size,
            train_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )
        val_dataloader = tr.make_dataloader(
            val_dataset,
            batch_size,
            val_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )
        print("DataLoaders initialized...")

        # Time each epoch and print to stdout
        startTime = time.time()

        # Train an Epoch
        tr.train_LRsched_loderunner_epoch(
            channel_map,
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            LRsched=LRsched,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            verbose=False,
        )

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...", flush=True)
        print("Epoch time (minutes):", epoch_time, flush=True)

        # Clear GPU memory after each epoch
        torch.cuda.empty_cache()

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        model, optimizer, epochIDX, new_h5_path, compiled=False
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
