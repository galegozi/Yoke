import argparse, os, torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from yoke.models.vit.swin.bomberman import DiffusionForecaster  # our new model
# from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet (if we want to use dataset for loading)
# But we'll manually load .npz files for the chosen simulation for precise control.

def load_simulation_sequence(data_dir, sim_prefix, max_frames=100, half_image=True):
    """Load all frames of a given simulation into a tensor [C, T, H, W]."""
    frames = []
    for idx in range(max_frames):
        fname = f"{sim_prefix}_pvi_idx{idx:05d}.npz"
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            break  # stop if file doesn't exist (end of sim)
        npz = np.load(fpath)
        # Assuming hydro_fields order same as training (density_case,...)
        fields = []
        for field in ["density_case","density_cushion","density_maincharge","density_outside_air",
                      "density_striker","density_throw","Uvelocity","Wvelocity"]:
            img = LSCread_npz_NaN(npz, field)
            if not half_image:
                img = np.concatenate((np.fliplr(img), img), axis=1)
            fields.append(img)
        npz.close()
        frame_array = np.stack(fields, axis=0)  # [C, H, W]
        frames.append(torch.tensor(frame_array, dtype=torch.float32))
    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded for simulation {sim_prefix}")
    # Stack along time dimension
    sim_tensor = torch.stack(frames, dim=1)  # shape [C, T, H, W]
    return sim_tensor

def generate_sequence(model: DiffusionForecaster, initial_frame: torch.Tensor, T: int, device='cpu'):
    """Generate a sequence of length T given the initial frame using the diffusion model."""
    model.eval()
    model.to(device)
    # Prepare an array for frames. We'll do iterative diffusion sampling:
    C, H, W = initial_frame.shape
    # Initialize with noise for frames 1..T-1
    generated = torch.zeros((C, T, H, W), device=device)
    generated[:,0,...] = initial_frame.to(device)  # set initial frame
    # Start with pure noise for future frames
    generated[:,1:,...] = torch.randn((C, T-1, H, W), device=device)
    generated = generated.unsqueeze(0)  # add batch dim [1, C, T, H, W]
    # Iterative denoising from high timestep down to 0
    diffusion_steps = 1000  # assume 1000 diffusion steps were used in training schedule
    for step in range(diffusion_steps, 0, -1):
        t = torch.tensor([step/1000.0], device=device)  # normalized timestep
        # Ensure initial frame remains fixed (conditioning):
        generated[:, :, 0, ...] = initial_frame.to(device)
        # Predict noise at this step
        with torch.no_grad():
            pred_noise = model(generated, None, None, t)  # model expects in_vars/out_vars internally or we set them earlier
        # Update the frames 1..T-1 using a simple DDPM update step (here a simplified version):
        # x_{t-1} = 1/alpha * (x_t - sigma * pred_noise) [not exact formula, just illustrative]
        # For correctness, use the actual diffusion schedule equations. We'll assume a linear schedule:
        alpha = 1 - (step/1000.0)  # dummy alpha, purely for illustration
        generated[:, :, 1:, ...] = (1/alpha) * (generated[:, :, 1:, ...] - (1-alpha)*pred_noise[:, :, 1:, ...])
    # After loop, generated should be an approximation of x_0 (predicted clean sequence)
    generated = generated.cpu().squeeze(0)  # [C, T, H, W]
    return generated

def plot_frame_comparison(true_frame, pred_frame, Rcoords, Zcoords, title, outpath):
    """Plot true vs predicted frame (density field) with discrepancy."""
    # Here we visualize total density as in original script – sum of first 6 channels (density_* fields)
    true_rho = true_frame[0:6, ...].sum(0)    # sum density channels
    pred_rho = pred_frame[0:6, ...].sum(0)
    discrepancy = torch.abs(true_rho - pred_rho)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(title, fontsize=18)
    im1 = ax1.imshow(true_rho, origin="lower", extent=[Rcoords.min(), Rcoords.max(), Zcoords.min(), Zcoords.max()],
                     cmap="jet")
    ax1.set_title("True"); ax1.set_xlabel("R"); ax1.set_ylabel("Z")
    im2 = ax2.imshow(pred_rho, origin="lower", extent=[Rcoords.min(), Rcoords.max(), Zcoords.min(), Zcoords.max()],
                     cmap="jet", vmin=true_rho.min().item(), vmax=true_rho.max().item())
    ax2.set_title("Predicted"); ax2.set_yticks([])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2).set_label("Density (g/cc)")
    im3 = ax3.imshow(discrepancy, origin="lower", extent=[Rcoords.min(), Rcoords.max(), Zcoords.min(), Zcoords.max()],
                     cmap="hot", vmax=0.5 * discrepancy.max().item())
    ax3.set_title("Discrepancy"); ax3.set_yticks([])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3).set_label("|True - Pred|")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animation of DiffusionForecaster on LSC simulation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt or .hdf5)")
    parser.add_argument("--sim_id", type=int, default=0, help="Simulation run ID to visualize (e.g., 201 for id00201)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing LSC .npz files")
    parser.add_argument("--out_dir", type=str, default="./diffusion_anime_output", help="Directory to save output images")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to generate/visualize")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sim_prefix = f"lsc240420_id{args.sim_id:05d}_pvi"  # assuming similar naming convention
    # Load the ground-truth sequence
    true_seq = load_simulation_sequence(args.data_dir, sim_prefix, max_frames=args.frames)
    initial_frame = true_seq[:, 0, ...]
    T = true_seq.shape[1]

    # Load model
    model = DiffusionForecaster(
        default_vars=[...],  # list of variable names as used in training
        image_size=true_seq.shape[2:4], seq_length=T, embed_dim=128)
    # Load model weights from checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # Generate predicted sequence via diffusion sampling
    pred_seq = generate_sequence(model, initial_frame, T, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Get coordinate arrays for plotting (assuming R and Z coordinates can be derived or are fixed)
    # For example, if each pixel = 0.1 units, etc. Here we use index as coordinate for simplicity:
    H, W = true_seq.shape[2], true_seq.shape[3]
    Rcoords = np.linspace(0, W-1, W)  # dummy coordinate array
    Zcoords = np.linspace(0, H-1, H)

    # Loop through frames and save comparison images
    for t in range(T):
        true_frame = true_seq[:, t, ...]
        pred_frame = pred_seq[:, t, ...]
        sim_time = 0.25 * t  # if 0.25us per frame, adjust as needed
        title = f"Frame {t} (T={sim_time:.2f} µs)"
        outfile = os.path.join(args.out_dir, f"diffusion_pred_id{args.sim_id:05d}_idx{t:05d}.png")
        plot_frame_comparison(true_frame, pred_frame, Rcoords, Zcoords, title, outfile)
    print(f"Saved visualization frames to {args.out_dir}")
