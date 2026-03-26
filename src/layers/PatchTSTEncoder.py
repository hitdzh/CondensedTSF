"""
PatchTST-based Feature Extractor for Microscopic Time Series Detail Capture

This module implements a modified PatchTST architecture specifically designed to capture
tiny, local implementation details (local mutations, high-frequency glitches, etc.) in
multivariate time series for K-Center greedy algorithm application.

Key Features:
    - Overlapping micro-patching (stride < patch_len) to preserve boundary details
    - No global average pooling (GAP) to avoid smoothing local anomalies
    - RevIN normalization to focus on morphological details
    - Optional RevIN parameter concatenation for scale-aware representation
    - L2 normalization for cosine distance compatibility in K-Center

Author: Modified PatchTST Implementation
Date: 2026-03-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)

    Normalizes each instance independently by subtracting mean and dividing by std.
    This forces the Transformer to learn pure morphological details rather than
    absolute scale information, which is crucial for capturing microscopic patterns.

    The affine parameters (mean, std) can be optionally concatenated to the final
    feature vector to make the representation scale-aware for K-Center.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        Parameters
        ----------
        num_features : int
            Number of input features (channels)
        eps : float
            Small constant for numerical stability
        affine : bool
            If True, uses learnable affine parameters (not recommended for this use case)
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply RevIN normalization

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features)

        Returns
        -------
        normalized : torch.Tensor
            Normalized tensor, same shape as input
        mean : torch.Tensor
            Mean values per instance and feature, shape (batch_size, 1, num_features)
        std : torch.Tensor
            Std values per instance and feature, shape (batch_size, 1, num_features)
        """
        batch_size, seq_len, num_features = x.shape

        # Compute mean and std along sequence dimension for each instance and feature
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, C)

        # Normalize
        normalized = (x - mean) / std

        # Apply learnable affine transformation if enabled
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias

        return normalized, mean, std

    def inverse(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor
        mean : torch.Tensor
            Original mean values
        std : torch.Tensor
            Original std values

        Returns
        -------
        torch.Tensor
            Denormalized tensor
        """
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        return x * std + mean


class Patching(nn.Module):
    """
    Overlapping Micro-Patching Layer

    Creates overlapping patches from input time series to preserve microscopic details
    at patch boundaries. This is critical for capturing local mutations and high-
    frequency glitches that would be lost with non-overlapping patches.

    Key Design:
        - Small patch_len (8 or 16) for fine-grained analysis
        - stride < patch_len (typically half) ensures overlapping patches
        - Overlap prevents information loss at patch boundaries
    """

    def __init__(self, patch_len: int = 16, stride: int = 8):
        """
        Parameters
        ----------
        patch_len : int
            Length of each patch (should be small, e.g., 8 or 16)
        stride : int
            Stride between patches (should be < patch_len for overlap)
        """
        super(Patching, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Note: We don't use nn.Unfold here because it requires 4D input
        # Instead, we'll implement custom patching in forward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to overlapping patches

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, c_in)

        Returns
        -------
        patches : torch.Tensor
            Patchified tensor of shape (batch_size, num_patches, patch_len, c_in)
        """
        batch_size, seq_len, c_in = x.shape

        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches using unfold
        # Reshape to (batch_size, c_in, seq_len) for unfold
        x = x.transpose(1, 2)  # (B, C, L)

        # Use unfold to create patches
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches shape: (batch_size, c_in, num_patches, patch_len)

        # Reshape to (batch_size, num_patches, patch_len, c_in)
        patches = patches.permute(0, 2, 3, 1).contiguous()

        return patches


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patches

    Unlike sinusoidal encoding, learnable encoding can adapt to the specific
    temporal patterns in the data during training.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Parameters
        ----------
        d_model : int
            Model dimension
        max_len : int
            Maximum sequence length (number of patches)
        """
        super(PositionalEncoding, self).__init__()

        # Create learnable position embeddings
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_patches, d_model)

        Returns
        -------
        torch.Tensor
            Tensor with positional encoding added
        """
        num_patches = x.size(1)
        return x + self.position_embedding[:num_patches, :].unsqueeze(0)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with multi-head self-attention

    Processes patchified time series to extract high-level features while
    preserving local details through overlapping patches.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Parameters
        ----------
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        e_layers : int
            Number of encoder layers
        d_ff : int
            Feed-forward network dimension
        dropout : float
            Dropout rate
        activation : str
            Activation function ('gelu' or 'relu')
        """
        super(TransformerEncoder, self).__init__()

        # Get activation function
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer Encoder

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_patches, d_model)

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, num_patches, d_model)
        """
        return self.encoder(x)


class AggregationHead(nn.Module):
    """
    Aggregation strategies for converting patch-level features to single vector

    CRITICAL: No Global Average Pooling (GAP) as it acts like a low-pass filter
    and smooths out local anomalies. Instead, we provide two strategies:

    1. Max Pooling: Extracts most significant local activation features
    2. Flatten + Linear: Preserves absolute temporal position of micro-features
    """

    def __init__(
        self,
        strategy: str = 'max',
        num_patches: int = None,
        d_model: int = None,
        target_dim: int = None
    ):
        """
        Parameters
        ----------
        strategy : str
            Aggregation strategy: 'max' or 'flatten'
        num_patches : int
            Number of patches (required for 'flatten' strategy)
        d_model : int
            Model dimension (required for 'flatten' strategy)
        target_dim : int
            Target output dimension (required for 'flatten' strategy)
        """
        super(AggregationHead, self).__init__()

        self.strategy = strategy

        if strategy == 'max':
            # Max pooling: extracts most salient local features
            # This preserves local anomalies that would be smoothed by average pooling
            pass  # No parameters needed

        elif strategy == 'flatten':
            # Flatten + Linear: preserves temporal position information
            # This is crucial for capturing WHEN micro-features occur
            if num_patches is None or d_model is None or target_dim is None:
                raise ValueError(
                    "num_patches, d_model, and target_dim must be provided for 'flatten' strategy"
                )

            self.projection = nn.Linear(num_patches * d_model, target_dim)

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch features to single vector

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_patches, d_model)

        Returns
        -------
        torch.Tensor
            Aggregated feature vector of shape (batch_size, d_model) or (batch_size, target_dim)
        """
        if self.strategy == 'max':
            # Max pooling over sequence dimension
            # This captures the most significant local activations across all patches
            return x.max(dim=1)[0]  # (batch_size, d_model)

        elif self.strategy == 'flatten':
            # Flatten and project to target dimension
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)  # (batch_size, num_patches * d_model)
            return self.projection(x_flat)  # (batch_size, target_dim)


class PatchTSTFeatureExtractor(nn.Module):
    """
    PatchTST-based Feature Extractor for Microscopic Time Series Detail Capture

    This model maps multivariate time series of length seq_len to a single
    high-dimensional feature vector, specifically designed to capture tiny
    and local implementation details (local mutations, high-frequency glitches)
    for use with K-Center greedy algorithm.

    Architecture:
        1. RevIN: Normalize to remove macro-level mean and variance
        2. Patching: Create overlapping patches (stride < patch_len)
        3. Projection: Linear mapping to d_model dimension
        4. Positional Encoding: Add learnable position information
        5. Transformer Encoder: Multi-layer self-attention processing
        6. Aggregation: Max pooling or Flatten+Linear
        7. RevIN Concatenation (optional): Append mean and std to output
        8. L2 Normalization: Normalize for cosine distance in K-Center

    Key Design Principles for Microscopic Detail Capture:
        - Overlapping patches prevent boundary information loss
        - No GAP (Global Average Pooling) to avoid smoothing anomalies
        - RevIN focuses learning on morphological patterns
        - Max pooling or flatten preserves spatial/temporal information
    """

    def __init__(
        self,
        c_in: int = 21,              # Input feature dimension
        seq_len: int = 336,          # Input sequence length
        patch_len: int = 16,         # Patch length (small for fine-grained analysis)
        stride: int = 8,             # Stride (must be < patch_len for overlap)
        d_model: int = 128,          # Model dimension
        n_heads: int = 4,            # Number of attention heads
        e_layers: int = 2,           # Number of encoder layers
        d_ff: int = 256,             # Feed-forward network dimension
        dropout: float = 0.1,        # Dropout rate
        activation: str = 'gelu',    # Activation function
        aggregation: str = 'max',    # Aggregation strategy: 'max' or 'flatten'
        target_dim: int = 256,       # Target output dimension (for 'flatten' strategy)
        concat_rev_params: bool = True  # Whether to concatenate RevIN parameters
    ):
        """
        Parameters
        ----------
        c_in : int
            Number of input features (channels)
        seq_len : int
            Input sequence length
        patch_len : int
            Length of each patch (should be small, e.g., 8 or 16)
        stride : int
            Stride between patches (should be < patch_len for overlap)
        d_model : int
            Transformer model dimension
        n_heads : int
            Number of attention heads
        e_layers : int
            Number of Transformer encoder layers
        d_ff : int
            Feed-forward network dimension
        dropout : float
            Dropout rate
        activation : str
            Activation function ('gelu' or 'relu')
        aggregation : str
            Aggregation strategy ('max' or 'flatten')
        target_dim : int
            Target output dimension (used when aggregation='flatten')
        concat_rev_params : bool
            Whether to concatenate RevIN mean/std to output vector
        """
        super(PatchTSTFeatureExtractor, self).__init__()

        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.aggregation = aggregation
        self.concat_rev_params = concat_rev_params

        # Validate parameters
        if stride >= patch_len:
            raise ValueError(f"stride ({stride}) must be < patch_len ({patch_len}) for overlapping patches")

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 1. RevIN: Reversible Instance Normalization
        # Removes macro-scale statistics to focus on morphological details
        self.revin = RevIN(num_features=c_in, eps=1e-5, affine=False)

        # 2. Patching: Create overlapping patches
        # Small patches with overlap preserve boundary details
        self.patching = Patching(patch_len=patch_len, stride=stride)

        # 3. Projection: Map each patch to d_model dimension
        # Input: (batch_size, num_patches, patch_len, c_in)
        # Output: (batch_size, num_patches, d_model)
        self.patch_projection = nn.Linear(patch_len * c_in, d_model)

        # 4. Positional Encoding: Learnable position embeddings
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=self.num_patches)

        # 5. Transformer Encoder: Multi-layer self-attention
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

        # 6. Aggregation Head: Convert patches to single vector
        # Determine output dimension based on aggregation strategy
        if aggregation == 'max':
            agg_output_dim = d_model
        elif aggregation == 'flatten':
            agg_output_dim = target_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self.aggregation_head = AggregationHead(
            strategy=aggregation,
            num_patches=self.num_patches,
            d_model=d_model,
            target_dim=target_dim
        )

        # 7. Optional: Concatenate RevIN parameters (mean and std)
        # This makes the representation scale-aware for K-Center
        if concat_rev_params:
            # Add 2 * c_in dimensions (mean and std for each channel)
            final_output_dim = agg_output_dim + 2 * c_in
        else:
            final_output_dim = agg_output_dim

        self.final_output_dim = final_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features from multivariate time series

        Parameters
        ----------
        x : torch.Tensor
            Input time series of shape (batch_size, seq_len, c_in)

        Returns
        -------
        features : torch.Tensor
            L2-normalized feature vector of shape (batch_size, final_output_dim)
            Ready for cosine distance computation in K-Center algorithm
        """
        batch_size = x.size(0)

        # Step 1: Apply RevIN normalization
        # This removes macro-level mean and variance, forcing the model to learn
        # pure morphological patterns and microscopic details
        x_normalized, revin_mean, revin_std = self.revin(x)
        # x_normalized: (batch_size, seq_len, c_in)
        # revin_mean, revin_std: (batch_size, 1, c_in)

        # Step 2: Create overlapping patches
        # Small patches with overlap prevent information loss at boundaries
        patches = self.patching(x_normalized)
        # patches: (batch_size, num_patches, patch_len, c_in)

        # Step 3: Project patches to d_model dimension
        patches_flat = patches.reshape(batch_size, self.num_patches, -1)
        # patches_flat: (batch_size, num_patches, patch_len * c_in)

        patch_embeddings = self.patch_projection(patches_flat)
        # patch_embeddings: (batch_size, num_patches, d_model)

        # Step 4: Add positional encoding
        patch_embeddings = self.pos_encoding(patch_embeddings)
        # patch_embeddings: (batch_size, num_patches, d_model)

        # Step 5: Process through Transformer Encoder
        # The self-attention mechanism captures dependencies between patches
        # while preserving local information within each patch
        encoded = self.transformer_encoder(patch_embeddings)
        # encoded: (batch_size, num_patches, d_model)

        # Step 6: Aggregate patch features to single vector
        # Using max pooling or flatten+linear to preserve local details
        features = self.aggregation_head(encoded)
        # features: (batch_size, d_model) or (batch_size, target_dim)

        # Step 7 (optional): Concatenate RevIN parameters
        # This makes the representation scale-aware, allowing K-Center to
        # distinguish between sequences with similar shapes but different scales
        if self.concat_rev_params:
            # Squeeze the time dimension and concatenate
            revin_mean = revin_mean.squeeze(1)  # (batch_size, c_in)
            revin_std = revin_std.squeeze(1)    # (batch_size, c_in)

            features = torch.cat([features, revin_mean, revin_std], dim=1)
            # features: (batch_size, agg_output_dim + 2 * c_in)

        # Step 8: L2 Normalization
        # Normalize feature vectors to unit length for cosine distance in K-Center
        # This is critical for stable distance computation
        features = F.normalize(features, p=2, dim=1)
        # features: (batch_size, final_output_dim), ||features||_2 = 1

        return features

    def get_output_dim(self) -> int:
        """
        Get the output feature dimension

        Returns
        -------
        int
            Output dimension of the feature vector
        """
        return self.final_output_dim


def create_patchtst_encoder(
    c_in: int = 21,
    seq_len: int = 336,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    e_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    activation: str = 'gelu',
    aggregation: str = 'max',
    target_dim: int = 256,
    concat_rev_params: bool = True,
    pretrained_path: str = None,
    device: str = 'cuda'
) -> PatchTSTFeatureExtractor:
    """
    Factory function to create a PatchTST Feature Extractor

    Parameters
    ----------
    c_in : int
        Number of input features
    seq_len : int
        Input sequence length
    patch_len : int
        Patch length (small for fine-grained analysis)
    stride : int
        Stride between patches (must be < patch_len)
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    e_layers : int
        Number of encoder layers
    d_ff : int
        Feed-forward network dimension
    dropout : float
        Dropout rate
    activation : str
        Activation function
    aggregation : str
        Aggregation strategy ('max' or 'flatten')
    target_dim : int
        Target output dimension (for 'flatten' strategy)
    concat_rev_params : bool
        Whether to concatenate RevIN parameters
    pretrained_path : str, optional
        Path to pretrained model weights
    device : str
        Device to load model on

    Returns
    -------
    encoder : PatchTSTFeatureExtractor
        Configured feature extractor
    """
    encoder = PatchTSTFeatureExtractor(
        c_in=c_in,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        aggregation=aggregation,
        target_dim=target_dim,
        concat_rev_params=concat_rev_params
    ).to(device)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        encoder.load_state_dict(state_dict)
        print(f"Loaded pretrained encoder from {pretrained_path}")

    encoder.eval()
    return encoder


if __name__ == '__main__':
    """
    Test script with dummy data to verify the implementation
    """
    print("=" * 80)
    print("PatchTST Feature Extractor - Test with Dummy Data")
    print("=" * 80)
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 4
    seq_len = 336
    c_in = 21

    print(f"Test Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Input features (c_in): {c_in}")
    print()

    # Create dummy input data
    # Simulating multivariate time series with some local anomalies
    x = torch.randn(batch_size, seq_len, c_in)

    # Add some artificial local anomalies (glitches)
    for i in range(batch_size):
        # Add a spike at a random location
        spike_loc = torch.randint(0, seq_len, (1,)).item()
        feature_idx = torch.randint(0, c_in, (1,)).item()
        x[i, spike_loc, feature_idx] += 5.0

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print()

    # Test different aggregation strategies
    for aggregation in ['max', 'flatten']:
        print("-" * 80)
        print(f"Testing with aggregation strategy: {aggregation.upper()}")
        print("-" * 80)

        # Create the model
        model = PatchTSTFeatureExtractor(
            c_in=c_in,
            seq_len=seq_len,
            patch_len=16,        # Small patch size for fine-grained analysis
            stride=8,            # 50% overlap (stride = patch_len / 2)
            d_model=128,
            n_heads=4,
            e_layers=2,
            d_ff=256,
            dropout=0.1,
            activation='gelu',
            aggregation=aggregation,
            target_dim=256,
            concat_rev_params=True
        )

        print(f"Model Architecture:")
        print(f"  - Patch length: {model.patch_len}")
        print(f"  - Stride: {model.stride}")
        print(f"  - Number of patches: {model.num_patches}")
        print(f"  - Output dimension: {model.get_output_dim()}")
        print(f"  - Aggregation strategy: {aggregation}")
        print(f"  - Concatenate RevIN params: {model.concat_rev_params}")
        print()

        # Forward pass
        model.eval()
        with torch.no_grad():
            features = model(x)

        print(f"Output shape: {features.shape}")
        print(f"Output dimension: {features.shape[-1]}")
        print(f"Output L2 norm (should be 1.0): {torch.norm(features[0], p=2).item():.6f}")
        print()

        # Verify L2 normalization
        l2_norms = torch.norm(features, p=2, dim=1)
        print(f"L2 norms for all samples: {l2_norms}")
        print(f"Max deviation from 1.0: {(l2_norms - 1.0).abs().max().item():.6f}")
        print()

        # Test feature consistency
        print("Feature Statistics:")
        print(f"  - Mean: {features.mean().item():.6f}")
        print(f"  - Std: {features.std().item():.6f}")
        print(f"  - Min: {features.min().item():.6f}")
        print(f"  - Max: {features.max().item():.6f}")
        print()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print()
    print("The PatchTST Feature Extractor is ready for use with K-Center algorithm.")
    print("The output features are L2-normalized and suitable for cosine distance computation.")
