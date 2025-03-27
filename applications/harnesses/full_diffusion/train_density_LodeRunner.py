import argparse, time, os
import torch
from torch import nn
from torch.utils.data import DataLoader
from yoke.models.vit.swin.bomberman import DiffusionForecaster
from yoke.datasets.lsc_dataset import LSC_rho2rho_sequential_DataSet

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train DiffusionForecaster on LSC data")
parser.add_argument("--seq_len", type=int, default=10, help="Number of frames per sample sequence (including initial)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
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
args = parser.parse_args()

ylogger.configure_logger("yoke_logger", level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

studyIDX = args.studyIDX

train_filelist = args.FILELIST_DIR + args.train_filelist
validation_filelist = args.FILELIST_DIR + args.validation_filelist
test_filelist = args.FILELIST_DIR + args.test_filelist

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

# Prepare datasets and loaders
train_dataset = LSC_rho2rho_sequential_DataSet(
    LSC_NPZ_DIR=args.LSC_NPZ_DIR,
    file_prefix_list=train_filelist,
    max_file_checks=10,
    half_image=True,
    seq_len=args.seq_len
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataset = LSC_rho2rho_sequential_DataSet(
    LSC_NPZ_DIR=args.LSC_NPZ_DIR,
    file_prefix_list=validation_filelist,
    max_file_checks=10,
    half_image=True,
    seq_len=args.seq_len
)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Initialize model, optimizer
default_vars = train_dataset.hydro_fields.tolist()  # list of variable names from dataset
# Image size (H,W) can be inferred from one sample
sample = train_dataset[0]  # (initial, seq) or seq
if isinstance(sample, tuple):
    _, seq_sample = sample
else:
    seq_sample = sample
_, T, H, W = seq_sample.shape  # seq_sample shape [C, T, H, W]
model = DiffusionForecaster(default_vars=default_vars, image_size=(H, W), seq_length=T, embed_dim=args.embed_dim)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

# Training loop
#TODO: load checkpoint if continuing and make compatible with slurm.
os.makedirs(args.checkpoint_dir, exist_ok=True)
for epoch in range(1, args.epochs+1):
    model.train()
    epoch_start = time.time()
    train_losses = []
    for batch in train_loader:
        # batch could be either (initial, seq) or just seq depending on dataset return
        if isinstance(batch, tuple):
            _, seq = batch  # we don't actually need initial separately since seq contains it at index 0
        else:
            seq = batch
        seq = seq.to(model.var_embed.device)  # move to GPU if available
        B, C, T, H, W = seq.shape
        # Sample random diffusion timestep for each sample in batch
        # Use a linear schedule on [0,1] for simplicity
        t_vals = torch.rand(B, device=seq.device)  # random uniform in [0,1)
        # Prepare noisy sequence
        noise = torch.randn_like(seq)
        # Keep initial frame noise-free
        noise[:, :, 0, ...] = 0.0
        noisy_seq = seq.clone()
        # Apply noise to frames 1..T-1: x_t = sqrt(alpha)*x0 + sqrt(1-alpha)*noise.
        # Let's say alpha = 1 - t (simple linear decay with t)
        alpha = 1 - t_vals.view(B, 1, 1, 1, 1)  # shape it for broadcasting
        noisy_seq[:, :, 1:, ...] = (alpha * seq[:, :, 1:, ...] + (1-alpha) * noise[:, :, 1:, ...])
        # Forward pass: predict noise
        diff_steps = t_vals  # using t (0-1) as diffusion step indicator
        pred_noise = model(noisy_seq, None, None, diff_steps)
        # Compute loss on the future frames (out_vars = all variables by default)
        target_noise = noise  # we want to predict the exact noise added
        loss = loss_fn(pred_noise, target_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    elapsed = time.time() - epoch_start

    # Validation loop (compute average loss or other metrics)
    avg_val_loss = None
    if val_loader:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch[1] if isinstance(batch, tuple) else batch
                seq = seq.to(model.var_embed.device)
                B, C, T, H, W = seq.shape
                t_vals = torch.rand(B, device=seq.device)
                noise = torch.randn_like(seq)
                noise[:, :, 0, ...] = 0.0
                noisy_seq = seq.clone()
                alpha = 1 - t_vals.view(B, 1, 1, 1, 1)
                noisy_seq[:, :, 1:, ...] = (alpha * seq[:, :, 1:, ...] + (1-alpha) * noise[:, :, 1:, ...])
                pred_noise = model(noisy_seq, None, None, t_vals)
                loss = loss_fn(pred_noise, noise)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
    # Print epoch results
    if avg_val_loss is not None:
        print(f"Epoch {epoch}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f} (Time {elapsed:.2f}s)")
    else:
        print(f"Epoch {epoch}/{args.epochs}: Train Loss = {avg_train_loss:.4f} (Time {elapsed:.2f}s)")

    # Save checkpoint periodically
    if epoch % 5 == 0 or epoch == args.epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"diffusion_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
