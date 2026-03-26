"""
Pretrain PatchTST Encoder using Masked Reconstruction

This script pretrains the PatchTST-based feature extractor using masked reconstruction.
The encoder learns to capture microscopic details in time series data, which is essential
for K-Center greedy algorithm application.

Pretraining Method:
    - Randomly mask patches of the input sequence
    - Use PatchTST encoder to extract features
    - Reconstruct masked patches using a lightweight decoder
    - Minimize reconstruction loss

Usage:
    python pretrain_encoder.py --epochs 30 --batch_size 32 --mask_ratio 0.75

    # Use pretrained encoder for data selection
    python get_condensed_dataset.py --algorithm kcenter --k 500 \\
        --pretrained_path checkpoints/patchtst_pretrained.pth
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loader import Dataset_Custom
from src.layers.PatchTSTEncoder import (
    PatchTSTFeatureExtractor,
    RevIN,
    Patching
)


class MaskedPatchTSTAutoencoder(nn.Module):
    """
    Masked Autoencoder for PatchTST Pretraining

    Architecture:
        1. Encoder: PatchTST Feature Extractor (without final L2 normalization)
        2. Decoder: Lightweight decoder to reconstruct masked patches

    The encoder learns to capture microscopic details while the decoder
    reconstructs masked patches, forcing the encoder to learn robust representations.
    """

    def __init__(
        self,
        c_in: int = 21,
        seq_len: int = 336,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
        aggregation: str = 'max'
    ):
        super(MaskedPatchTSTAutoencoder, self).__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.aggregation = aggregation

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # RevIN normalization
        self.revin = RevIN(num_features=c_in, eps=1e-5, affine=False)

        # Patching
        self.patching = Patching(patch_len=patch_len, stride=stride)

        # Patch projection
        self.patch_projection = nn.Linear(patch_len * c_in, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Decoder: reconstruct masked patches
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, patch_len * c_in)
        )

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def random_mask(self, batch_size: int, device: torch.Tensor) -> torch.Tensor:
        """
        Generate random mask for patches

        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.Tensor
            Device

        Returns
        -------
        mask_indices : torch.Tensor
            Boolean mask, True indicates masked patches
            Shape: (batch_size, num_patches)
        """
        # Calculate number of patches to mask
        n_masked = int(self.num_patches * self.mask_ratio)

        # Create random mask for each sample
        mask_indices = []
        for _ in range(batch_size):
            indices = torch.randperm(self.num_patches, device=device)
            mask = torch.zeros(self.num_patches, dtype=torch.bool, device=device)
            mask[indices[:n_masked]] = True
            mask_indices.append(mask)

        mask_indices = torch.stack(mask_indices)  # (B, num_patches)
        return mask_indices

    def forward(self, x: torch.Tensor, return_reconstruction: bool = True) -> tuple:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch_size, seq_len, c_in)
        return_reconstruction : bool
            Whether to return reconstruction

        Returns
        -------
        features : torch.Tensor
            Feature vectors (if return_reconstruction=False)
        reconstruction : torch.Tensor
            Reconstructed patches (if return_reconstruction=True)
        mask_indices : torch.Tensor
            Mask indices (if return_reconstruction=True)
        """
        batch_size = x.size(0)

        # Apply RevIN normalization
        x_normalized, revin_mean, revin_std = self.revin(x)
        # x_normalized: (B, L, C)

        # Create patches
        patches = self.patching(x_normalized)
        # patches: (B, num_patches, patch_len, C)

        # Save original patches for reconstruction loss
        patches_flat = patches.reshape(batch_size, self.num_patches, -1)
        # patches_flat: (B, num_patches, patch_len * C)

        # Project patches to d_model
        patch_embeddings = self.patch_projection(patches_flat)
        # patch_embeddings: (B, num_patches, d_model)

        if return_reconstruction:
            # Generate mask
            mask_indices = self.random_mask(batch_size, x.device)
            # mask_indices: (B, num_patches), True = masked

            # Add mask tokens at masked positions
            masked_embeddings = patch_embeddings.clone()
            masked_embeddings[mask_indices] = self.mask_token.squeeze(0)

            # Add positional encoding
            masked_embeddings = masked_embeddings + self.pos_encoding

            # Process through Transformer
            encoded = self.transformer_encoder(masked_embeddings)
            # encoded: (B, num_patches, d_model)

            # Decode to reconstruct patches
            reconstructed_patches = self.decoder(encoded)
            # reconstructed_patches: (B, num_patches, patch_len * C)

            return encoded, reconstructed_patches, mask_indices, patches_flat

        else:
            # Inference mode: extract features without masking
            # Add positional encoding
            patch_embeddings = patch_embeddings + self.pos_encoding

            # Process through Transformer
            encoded = self.transformer_encoder(patch_embeddings)
            # encoded: (B, num_patches, d_model)

            # Aggregate to single vector
            if self.aggregation == 'max':
                features = encoded.max(dim=1)[0]  # (B, d_model)
            elif self.aggregation == 'flatten':
                features = encoded.reshape(batch_size, -1)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            # Concatenate RevIN parameters
            revin_mean = revin_mean.squeeze(1)  # (B, C)
            revin_std = revin_std.squeeze(1)    # (B, C)
            features = torch.cat([features, revin_mean, revin_std], dim=1)

            # L2 normalize
            features = nn.functional.normalize(features, p=2, dim=1)

            return features


def pretrain_model(model, train_loader, val_loader, args, device):
    """Pretrain PatchTST model"""

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print("\nStart pretraining...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_x, _, batch_x_mark, _ in pbar:
            batch_x = batch_x.float().to(device)

            # Forward pass
            encoded, reconstructed, mask_indices, original_patches = model(
                batch_x,
                return_reconstruction=True
            )

            # Compute reconstruction loss only on masked patches
            batch_size = batch_x.size(0)

            # Extract masked patches
            n_masked = mask_indices.sum(dim=1)  # (B,)

            # Compute loss for each sample
            total_loss = 0
            valid_samples = 0

            for i in range(batch_size):
                if n_masked[i] > 0:
                    # Get masked patch indices
                    masked_idx = mask_indices[i]  # (num_patches,)

                    # Extract reconstructed and original masked patches
                    pred_masked = reconstructed[i][masked_idx]  # (n_masked, patch_len*C)
                    true_masked = original_patches[i][masked_idx]  # (n_masked, patch_len*C)

                    # Compute MSE loss
                    loss = criterion(pred_masked, true_masked)
                    total_loss += loss
                    valid_samples += 1

            if valid_samples > 0:
                loss = total_loss / valid_samples
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss = np.mean(train_loss) if train_loss else 0.0

        # Validation
        model.eval()
        val_loss = []

        with torch.no_grad():
            for batch_x, _, batch_x_mark, _ in val_loader:
                batch_x = batch_x.float().to(device)

                encoded, reconstructed, mask_indices, original_patches = model(
                    batch_x,
                    return_reconstruction=True
                )

                batch_size = batch_x.size(0)
                n_masked = mask_indices.sum(dim=1)

                total_loss = 0
                valid_samples = 0

                for i in range(batch_size):
                    if n_masked[i] > 0:
                        masked_idx = mask_indices[i]
                        pred_masked = reconstructed[i][masked_idx]
                        true_masked = original_patches[i][masked_idx]

                        loss = criterion(pred_masked, true_masked)
                        total_loss += loss
                        valid_samples += 1

                if valid_samples > 0:
                    loss = total_loss / valid_samples
                    val_loss.append(loss.item())

        val_loss = np.mean(val_loss) if val_loss else 0.0

        # Update learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, args.save_path)
            print(f"  [Best model saved] Val Loss: {val_loss:.6f}")

    print("\nPretraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.save_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Pretrain PatchTST Encoder')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='dataset/weather.csv',
                        help='Dataset path')
    parser.add_argument('--root_path', type=str, default='./',
                        help='Root path')
    parser.add_argument('--seq_len', type=int, default=336,
                        help='Sequence length')
    parser.add_argument('--label_len', type=int, default=96,
                        help='Label length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting mode')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target column')
    parser.add_argument('--freq', type=str, default='10min',
                        help='Time frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # PatchTST parameters
    parser.add_argument('--c_in', type=int, default=21,
                        help='Input feature dimension')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length (small for fine-grained analysis)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride (must be < patch_len for overlap)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--aggregation', type=str, default='max',
                        choices=['max', 'flatten'],
                        help='Aggregation strategy')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio (0-1)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device')

    # Save parameters
    parser.add_argument('--save_path', type=str,
                        default='outputs/checkpoints/patchtst_pretrained.pth',
                        help='Save path')

    # Resume parameters
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a pretrained checkpoint (skip training, extract encoder only)')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Device configuration
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    print("Pretrain PatchTST Encoder for Time Series")
    print(f"Dataset: {args.data_path}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Patch configuration: patch_len={args.patch_len}, stride={args.stride}")
    print(f"Model dimension: {args.d_model}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Training epochs: {args.epochs}")
    print(f"Device: {device}")
    print()

    # Validate parameters
    if args.stride >= args.patch_len:
        raise ValueError(f"stride ({args.stride}) must be < patch_len ({args.patch_len}) for overlapping patches")

    # Load dataset
    print("Loading dataset...")

    train_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=0,
        freq=args.freq
    )

    val_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=0,
        freq=args.freq
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print()

    # Create model
    print("Creating PatchTST Masked Autoencoder...")

    model = MaskedPatchTSTAutoencoder(
        c_in=args.c_in,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        aggregation=args.aggregation
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Check if resuming from checkpoint
    if args.resume_from:
        print()
        print(f"Resume from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model from full checkpoint")
    else:
        print()
        # Pretrain
        pretrain_model(model, train_loader, val_loader, args, device)

    # Extract and save encoder-only weights
    print()
    print("Extracting and saving encoder-only weights...")

    # Create a PatchTSTFeatureExtractor with the same configuration
    from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor

    encoder_only = PatchTSTFeatureExtractor(
        c_in=args.c_in,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=0.0,  # No dropout for inference
        activation='gelu',
        aggregation=args.aggregation,
        target_dim=256,
        concat_rev_params=True
    ).to(device)

    # Copy weights from pretrained model to encoder
    encoder_only.revin.load_state_dict(model.revin.state_dict())
    encoder_only.patching.patch_len = model.patch_len
    encoder_only.patching.stride = model.stride
    encoder_only.patch_projection.load_state_dict(model.patch_projection.state_dict())
    # pos_encoding: model uses Parameter (1, num_patches, d_model), encoder uses module (max_len, d_model)
    with torch.no_grad():
        encoder_only.pos_encoding.position_embedding[:model.num_patches].copy_(model.pos_encoding.squeeze(0))

    # TransformerEncoder key mapping: model uses nn.TransformerEncoder, encoder uses custom TransformerEncoder wrapper
    # Source keys: "layers.0...", "norm..."
    # Target keys: "encoder.layers.0...", "encoder.norm..."
    state_dict = model.transformer_encoder.state_dict()
    remapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'encoder.' + key
        remapped_state_dict[new_key] = value
    encoder_only.transformer_encoder.load_state_dict(remapped_state_dict)

    # Note: aggregation weights (max pooling or flatten reshape) are not applicable here
    # since MaskedPatchTSTAutoencoder handles aggregation differently

    # Save encoder-only weights with numbered ID
    save_dir = os.path.dirname(args.save_path)
    base_name = os.path.basename(args.save_path).replace('.pth', '')
    os.makedirs(save_dir, exist_ok=True)

    # Find existing encoder files and determine next ID
    existing_ids = []
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            if f.startswith(f'{base_name}_encoder_only_') and f.endswith('.pth'):
                # Extract ID from filename like: base_encoder_only_001.pth
                try:
                    id_part = f.replace(f'{base_name}_encoder_only_', '').replace('.pth', '')
                    existing_ids.append(int(id_part))
                except ValueError:
                    continue

    next_id = max(existing_ids) + 1 if existing_ids else 1
    encoder_save_path = os.path.join(save_dir, f'{base_name}_encoder_only_{next_id:03d}.pth')
    torch.save(encoder_only.state_dict(), encoder_save_path)

    print(f"Encoder-only weights saved to: {encoder_save_path}")

    print()
    print("Pretraining completed successfully!")
    print()
    print("To use the pretrained encoder for data selection:")
    print(f"  python scripts/get_condensed_dataset.py \\")
    print(f"    --algorithm kcenter \\")
    print(f"    --k 500 \\")
    print(f"    --pretrained_path {encoder_save_path}")
    print()


if __name__ == '__main__':
    main()
