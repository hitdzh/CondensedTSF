"""
Get Condensed Dataset Script using PatchTST Encoder

Uses PatchTST-based feature extractor with K-Center or Herding algorithm
to select representative samples from the original dataset and save them.

Usage:
    # Use K-Center algorithm
    python get_condensed_dataset.py --algorithm kcenter --k 500

    # Use Herding algorithm
    python get_condensed_dataset.py --algorithm herding --k 1000

    # Custom parameters
    python get_condensed_dataset.py \\
        --algorithm kcenter \\
        --k 500 \\
        --seq_len 336 \\
        --pred_len 96 \\
        --patch_len 16 \\
        --stride 8 \\
        --d_model 128 \\
        --data_path dataset/weather.csv \\
        --save_dir condensed_datasets/weather_kcenter_k500
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.data_loader import Dataset_Custom
from src.layers.PatchTSTEncoder import create_patchtst_encoder
from src.utils.feature_selection import select_samples, save_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Get Condensed Dataset')

    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='kcenter',
                        choices=['kcenter', 'herding'],
                        help='Data selection algorithm: kcenter or herding')
    parser.add_argument('--k', type=int, default=500,
                        help='Number of samples to select')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='dataset/weather.csv',
                        help='Dataset path')
    parser.add_argument('--root_path', type=str, default='./',
                        help='Root path')
    parser.add_argument('--seq_len', type=int, default=336,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=96,
                        help='Label sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting mode: M=multivariate, S=univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target column')
    parser.add_argument('--freq', type=str, default='10min',
                        help='Time frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Data loading batch size')

    # PatchTST Encoder parameters
    parser.add_argument('--c_in', type=int, default=21,
                        help='Input feature dimension')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length (small for fine-grained analysis)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride (must be < patch_len for overlapping patches)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer feature dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward network dimension')
    parser.add_argument('--aggregation', type=str, default='max',
                        choices=['max', 'flatten'],
                        help='Aggregation strategy: max or flatten')
    parser.add_argument('--target_dim', type=int, default=256,
                        help='Target output dimension (for flatten aggregation)')
    parser.add_argument('--concat_rev_params', type=bool, default=True,
                        help='Whether to concatenate RevIN parameters')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Pretrained encoder path (optional)')

    # Save parameters
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory (default: condensed_datasets/{data}_{algorithm}_k{k})')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Print configuration
    print("Get Condensed Dataset - {} Algorithm".format(args.algorithm.upper()))
    print(f"Algorithm: {args.algorithm}")
    print(f"Samples to select (k): {args.k}")
    print(f"Dataset: {args.data_path}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Feature dimension: {args.d_model}")
    print(f"Random seed: {args.seed}")
    print()

    # ========== 1. Load dataset ==========
    print("Step 1: Loading dataset")

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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training set size: {len(train_set)}")
    print(f"Data shape: seq_x={train_set.data_x.shape}")
    print()

    # ========== 2. Create PatchTST Encoder ==========
    print("Step 2: Create PatchTST Encoder")

    # Validate parameters
    if args.stride >= args.patch_len:
        raise ValueError(f"stride ({args.stride}) must be < patch_len ({args.patch_len}) for overlapping patches")

    encoder = create_patchtst_encoder(
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
        target_dim=args.target_dim,
        concat_rev_params=args.concat_rev_params,
        pretrained_path=args.pretrained_path,
        device=device
    )

    print(f"PatchTST Encoder Configuration:")
    print(f"  - Input dimension: {args.c_in}")
    print(f"  - Patch length: {args.patch_len}")
    print(f"  - Stride: {args.stride}")
    print(f"  - Output dimension: {encoder.get_output_dim()}")
    print(f"  - Attention heads: {args.n_heads}")
    print(f"  - Encoder layers: {args.e_layers}")
    print(f"  - FFN dimension: {args.d_ff}")
    print(f"  - Aggregation strategy: {args.aggregation}")
    print(f"  - Concat RevIN params: {args.concat_rev_params}")
    if args.pretrained_path:
        print(f"  - Pretrained weights: {args.pretrained_path}")
    print()

    # ========== 3. Test encoder ==========
    print("Step 3: Test encoder")

    batch_x, _, batch_x_mark, _ = next(iter(train_loader))
    batch_x = batch_x[:4].to(device)

    with torch.no_grad():
        features = encoder(batch_x)

    print(f"Input shape: {batch_x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {features.shape[-1]}")
    print()

    # ========== 4. Execute data selection ==========
    print("Step 4: Execute data selection")

    selected_indices, radius = select_samples(
        train_loader,
        encoder,
        k=args.k,
        algorithm=args.algorithm,
        device=device
    )

    print()
    print(f"{args.algorithm.upper()} Results:")
    print(f"  - Selected samples: {len(selected_indices)}")
    print(f"  - Coverage radius: {radius:.4f}")
    print(f"  - Compression ratio: {100 * len(selected_indices) / len(train_set):.2f}%")
    print()

    # ========== 5. Save condensed dataset ==========
    print("Step 5: Save condensed dataset")

    # Generate save directory
    if args.save_dir is None:
        data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        args.save_dir = f'condensed_datasets/{data_name}_{args.algorithm}_k{args.k}'

    # Prepare metadata
    metadata = {
        'algorithm': args.algorithm,
        'method': 'feature_based',
        'encoder_type': 'patchtst',
        'c_in': args.c_in,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'd_ff': args.d_ff,
        'aggregation': args.aggregation,
        'target_dim': args.target_dim,
        'concat_rev_params': args.concat_rev_params,
        'radius': radius,
        'k': args.k,
        'seq_len': args.seq_len,
        'label_len': args.label_len,
        'pred_len': args.pred_len,
        'features': args.features,
        'target': args.target,
        'freq': args.freq,
        'original_size': len(train_set),
        'condensed_size': len(selected_indices),
        'compression_ratio': 100 * len(selected_indices) / len(train_set),
        'seed': args.seed,
        'pretrained_path': args.pretrained_path,
    }

    save_path = save_dataset(
        train_loader,
        selected_indices,
        args.save_dir,
        metadata
    )

    print()
    print("Completed!")
    print(f"Condensed dataset saved to: {save_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Samples: {len(selected_indices)}")
    print(f"Coverage radius: {radius:.4f}")
    print(f"Compression ratio: {100 * len(selected_indices) / len(train_set):.2f}%")


if __name__ == '__main__':
    main()
