import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr

# Define a global variable for the number of channels
NUM_CHANNELS = 8

# Define the model architecture
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

    # Initialize model, optimizer, and loss function
    model = UNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            start_img, end_img, Dt = batch
            start_img, end_img = start_img.to(device), end_img.to(device)

            # Add noise to the clean images
            start_img_noisy = add_noise(start_img)
            end_img_noisy = add_noise(end_img)

            optimizer.zero_grad()
            pred_start_img = model(start_img_noisy)
            pred_end_img = model(end_img_noisy)
            loss_start = loss_fn(pred_start_img, start_img)
            loss_end = loss_fn(pred_end_img, end_img)
            loss = loss_start + loss_end
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(dataloader)}")

        # Save the model checkpoint at the end of each epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(dataloader),
        }, checkpoint_path)

        # Submit a new SLURM job for the next epoch
        next_epoch = epoch + 2
        if next_epoch <= args.epochs:
            slurm_command = f"sbatch --export=ALL,EPOCH={next_epoch} {args.slurm_script}"
            os.system(slurm_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model on LSC data")
    parser.add_argument("--LSC_NPZ_DIR", type=str, required=True, help="Directory containing LSC NPZ files")
    parser.add_argument("--file_prefix_list", type=str, required=True, help="File listing unique prefixes for simulations")
    parser.add_argument("--max_timeIDX_offset", type=int, default=3, help="Maximum time index offset")
    parser.add_argument("--max_file_checks", type=int, default=5, help="Maximum number of file checks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save the model checkpoints")
    parser.add_argument("--slurm_script", type=str, required=True, help="Path to the SLURM script for the next epoch")
    args = parser.parse_args()

    main(args)
