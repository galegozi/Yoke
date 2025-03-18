import os
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = LSC_rho2rho_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=args.file_prefix_list,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=args.max_file_checks,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize models
    loderunner_model = LodeRunner(
        default_vars=["density_case", "density_cushion", "density_maincharge", "density_outside_air", "density_striker", "density_throw", "Uvelocity", "Wvelocity"],
        image_size=(1120, 400),
        patch_size=(10, 5),
        embed_dim=512,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 11, 2),
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
    ).to(device)
    diffusion_model = UNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)

    # Load model checkpoints
    loderunner_checkpoint = torch.load(args.loderunner_checkpoint)
    loderunner_model.load_state_dict(loderunner_checkpoint['model_state_dict'])
    diffusion_checkpoint = torch.load(args.diffusion_checkpoint)
    diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])

    loderunner_model.eval()
    diffusion_model.eval()

    for batch in dataloader:
        start_img, end_img, Dt = batch
        start_img, end_img = start_img.to(device), end_img.to(device)

        with torch.no_grad():
            loderunner_pred = loderunner_model(start_img)
            diffusion_pred = diffusion_model(loderunner_pred)

        # Plot Truth/Prediction/Discrepancy panel.
        fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 6))
        fig1.suptitle(f"Time={Dt[0].item():.3f}us", fontsize=18)
        img1 = ax1.imshow(
            start_img[0].cpu().numpy().transpose(1, 2, 0),
            aspect="equal",
            origin="lower",
            cmap="jet",
            vmin=start_img.min().cpu().item(),
            vmax=start_img.max().cpu().item(),
        )
        ax1.set_ylabel("Z-axis", fontsize=16)
        ax1.set_xlabel("R-axis", fontsize=16)
        ax1.set_title("True Start Image", fontsize=18)

        img2 = ax2.imshow(
            end_img[0].cpu().numpy().transpose(1, 2, 0),
            aspect="equal",
            origin="lower",
            cmap="jet",
            vmin=end_img.min().cpu().item(),
            vmax=end_img.max().cpu().item(),
        )
        ax2.set_title("True End Image", fontsize=18)
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

        img3 = ax3.imshow(
            diffusion_pred[0].cpu().numpy().transpose(1, 2, 0),
            aspect="equal",
            origin="lower",
            cmap="jet",
            vmin=diffusion_pred.min().cpu().item(),
            vmax=diffusion_pred.max().cpu().item(),
        )
        ax3.set_title("Predicted Image", fontsize=18)
        ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

        discrepancy_loderunner = np.abs(start_img[0].cpu().numpy().transpose(1, 2, 0) - loderunner_pred[0].cpu().numpy().transpose(1, 2, 0))
        img4 = ax4.imshow(
            discrepancy_loderunner,
            aspect="equal",
            origin="lower",
            cmap="hot",
            vmin=discrepancy_loderunner.min(),
            vmax=0.3 * discrepancy_loderunner.max(),
        )
        ax4.set_title("Discrepancy between LodeRunner and True", fontsize=18)
        ax4.tick_params(axis="y", which="both", left=False, labelleft=False)

        discrepancy_diffusion = np.abs(end_img[0].cpu().numpy().transpose(1, 2, 0) - diffusion_pred[0].cpu().numpy().transpose(1, 2, 0))
        img5 = ax5.imshow(
            discrepancy_diffusion,
            aspect="equal",
            origin="lower",
            cmap="hot",
            vmin=discrepancy_diffusion.min(),
            vmax=0.3 * discrepancy_diffusion.max(),
        )
        ax5.set_title("Discrepancy between Diffusion and True", fontsize=18)
        ax5.tick_params(axis="y", which="both", left=False, labelleft=False)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with LodeRunner and Diffusion models")
    parser.add_argument("--LSC_NPZ_DIR", type=str, required=True, help="Directory containing LSC NPZ files")
    parser.add_argument("--file_prefix_list", type=str, required=True, help="File listing unique prefixes for simulations")
    parser.add_argument("--max_timeIDX_offset", type=int, default=3, help="Maximum time index offset")
    parser.add_argument("--max_file_checks", type=int, default=5, help="Maximum number of file checks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--loderunner_checkpoint", type=str, required=True, help="Path to LodeRunner model checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True, help="Path to Diffusion model checkpoint")
    args = parser.parse_args()

    main(args)
