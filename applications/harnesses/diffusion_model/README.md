# Diffusion Model Harness

This directory contains a harness for training a diffusion model as described in the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

## Purpose

The purpose of this harness is to provide a framework for training a diffusion model on the LSC dataset. The model architecture includes self-attention and UNet components, and the training process involves adding noise to the images and optimizing the model to denoise them.

## Usage

To use this harness, follow these steps:

1. **Prepare the dataset**: Ensure that the LSC dataset is available in the specified directory.
2. **Set up the environment**: Install the required dependencies and set up the environment.
3. **Run the training script**: Execute the training script with the appropriate arguments.

### Example

Here is an example of how to run the training script:

```bash
python train_diffusion_model.py \
    --LSC_NPZ_DIR /path/to/lsc_npz_files \
    --file_prefix_list /path/to/file_prefix_list.txt \
    --max_timeIDX_offset 3 \
    --max_file_checks 5 \
    --batch_size 8 \
    --num_workers 4 \
    --learning_rate 1e-3 \
    --epochs 50 \
    --checkpoint_dir /path/to/checkpoints \
    --slurm_script /path/to/slurm_script.sh
```

## Arguments

- `--LSC_NPZ_DIR`: Directory containing LSC NPZ files.
- `--file_prefix_list`: File listing unique prefixes for simulations.
- `--max_timeIDX_offset`: Maximum time index offset.
- `--max_file_checks`: Maximum number of file checks.
- `--batch_size`: Batch size for training.
- `--num_workers`: Number of workers for data loading.
- `--learning_rate`: Learning rate for optimizer.
- `--epochs`: Number of training epochs.
- `--checkpoint_dir`: Directory to save the model checkpoints.
- `--slurm_script`: Path to the SLURM script for the next epoch.
