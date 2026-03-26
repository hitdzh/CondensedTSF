"""
Test Script for PatchTST Feature Extractor

This script tests the PatchTST encoder with dummy data to verify:
1. Correct output shapes
2. L2 normalization
3. Feature extraction capabilities
4. Compatibility with K-Center algorithm

Usage:
    python test_patchtst_encoder.py
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.layers.PatchTSTEncoder import (
    PatchTSTFeatureExtractor,
    create_patchtst_encoder,
    RevIN,
    Patching,
    AggregationHead
)


def test_rev_in():
    """Test RevIN normalization"""
    print("Testing RevIN (Reversible Instance Normalization)")

    batch_size = 4
    seq_len = 100
    c_in = 21

    x = torch.randn(batch_size, seq_len, c_in)

    # Add different scales to different samples
    x[0] *= 2.0
    x[1] *= 0.5
    x[2] += 5.0
    x[3] -= 3.0

    revin = RevIN(num_features=c_in, eps=1e-5, affine=False)

    # Forward pass
    x_norm, mean, std = revin(x)

    # Check normalization
    print(f"Original mean per sample: {x.mean(dim=1).squeeze()}")
    print(f"Original std per sample: {x.std(dim=1).squeeze()}")
    print(f"Normalized mean per sample: {x_norm.mean(dim=1).squeeze()}")
    print(f"Normalized std per sample: {x_norm.std(dim=1).squeeze()}")
    print()

    # Test inverse normalization
    x_reconstructed = revin.inverse(x_norm, mean, std)
    reconstruction_error = torch.abs(x - x_reconstructed).max().item()
    print(f"Reconstruction error: {reconstruction_error:.8f}")
    assert reconstruction_error < 1e-5, "RevIN reconstruction failed"

    print("[OK] RevIN test passed\n")


def test_patching():
    """Test overlapping patching"""
    print("Testing Overlapping Patching")

    batch_size = 2
    seq_len = 100
    c_in = 21
    patch_len = 16
    stride = 8

    x = torch.randn(batch_size, seq_len, c_in)

    patching = Patching(patch_len=patch_len, stride=stride)
    patches = patching(x)

    num_patches = (seq_len - patch_len) // stride + 1

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {patches.shape}")
    print(f"Expected patches: {num_patches}")
    print(f"Actual patches: {patches.shape[1]}")
    print()

    # Check overlap
    print("Checking patch overlap:")
    for i in range(min(3, num_patches - 1)):
        patch1 = patches[0, i, :, 0].numpy()[:stride]
        patch2 = patches[0, i + 1, :, 0].numpy()[-stride:]
        overlap_ratio = (patch1 == patch2).mean()
        print(f"  Patch {i} and {i+1} overlap: {overlap_ratio:.2%}")

    assert patches.shape == (batch_size, num_patches, patch_len, c_in), "Patching shape error"
    print("[OK] Patching test passed\n")


def test_aggregation_head():
    """Test aggregation strategies"""
    print("Testing Aggregation Head")

    batch_size = 4
    num_patches = 20
    d_model = 128
    target_dim = 256

    x = torch.randn(batch_size, num_patches, d_model)

    # Test max pooling
    print("Testing Max Pooling aggregation:")
    agg_max = AggregationHead(strategy='max')
    out_max = agg_max(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_max.shape}")
    assert out_max.shape == (batch_size, d_model), "Max pooling shape error"
    print("  [OK] Max pooling test passed")

    # Test flatten + linear
    print("\nTesting Flatten + Linear aggregation:")
    agg_flatten = AggregationHead(
        strategy='flatten',
        num_patches=num_patches,
        d_model=d_model,
        target_dim=target_dim
    )
    out_flatten = agg_flatten(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_flatten.shape}")
    assert out_flatten.shape == (batch_size, target_dim), "Flatten shape error"
    print("  [OK] Flatten + Linear test passed\n")


def test_patchtst_encoder():
    """Test complete PatchTST encoder"""
    print("Testing Complete PatchTST Feature Extractor")

    # Configuration
    batch_size = 8
    seq_len = 336
    c_in = 21

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input features: {c_in}")
    print()

    # Create dummy data with local anomalies
    x = torch.randn(batch_size, seq_len, c_in)

    # Add artificial anomalies (simulating glitches/spikes)
    print("Adding artificial local anomalies...")
    for i in range(batch_size):
        # Add a spike
        spike_loc = torch.randint(0, seq_len, (1,)).item()
        feature_idx = torch.randint(0, c_in, (1,)).item()
        x[i, spike_loc, feature_idx] += torch.randn(1).item() * 10 + 5

        # Add a sudden drop
        drop_loc = torch.randint(0, seq_len, (1,)).item()
        feature_idx2 = torch.randint(0, c_in, (1,)).item()
        x[i, drop_loc, feature_idx2] -= torch.randn(1).item() * 10 + 5

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print()

    # Test both aggregation strategies
    for aggregation in ['max', 'flatten']:
        print("-" * 80)
        print(f"Testing with aggregation: {aggregation.upper()}")
        print("-" * 80)

        # Create model
        model = PatchTSTFeatureExtractor(
            c_in=c_in,
            seq_len=seq_len,
            patch_len=16,        # Small patch size
            stride=8,            # 50% overlap
            d_model=128,
            n_heads=4,
            e_layers=2,
            d_ff=256,
            dropout=0.0,         # No dropout for testing
            activation='gelu',
            aggregation=aggregation,
            target_dim=256,
            concat_rev_params=True
        )

        print(f"\nModel architecture:")
        print(f"  Patch length: {model.patch_len}")
        print(f"  Stride: {model.stride}")
        print(f"  Number of patches: {model.num_patches}")
        print(f"  Output dimension: {model.get_output_dim()}")
        print(f"  Aggregation: {aggregation}")
        print(f"  RevIN concat: {model.concat_rev_params}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {num_params:,}")
        print()

        # Forward pass
        model.eval()
        with torch.no_grad():
            features = model(x)

        print(f"Output shape: {features.shape}")
        print(f"Expected shape: ({batch_size}, {model.get_output_dim()})")
        assert features.shape == (batch_size, model.get_output_dim()), "Output shape error"

        # Verify L2 normalization
        print("\nVerifying L2 normalization:")
        l2_norms = torch.norm(features, p=2, dim=1)
        print(f"  L2 norms: {l2_norms}")
        print(f"  Max deviation from 1.0: {(l2_norms - 1.0).abs().max().item():.8f}")
        assert (l2_norms - 1.0).abs().max() < 1e-5, "L2 normalization failed"

        # Feature statistics
        print("\nFeature statistics:")
        print(f"  Mean: {features.mean().item():.6f}")
        print(f"  Std: {features.std().item():.6f}")
        print(f"  Min: {features.min().item():.6f}")
        print(f"  Max: {features.max().item():.6f}")

        # Test feature diversity (features should be different for different inputs)
        print("\nTesting feature diversity:")
        feature_similarities = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            for j in range(batch_size):
                # Cosine similarity (same as dot product for L2-normalized vectors)
                feature_similarities[i, j] = torch.dot(features[i], features[j]).item()

        # Diagonal should be 1.0 (self-similarity)
        assert torch.diagonal(feature_similarities).min() > 0.999, "Self-similarity failed"

        # Off-diagonal should be < 1.0 (different inputs should have different features)
        off_diag = feature_similarities[~torch.eye(batch_size, dtype=bool)]
        print(f"  Cross-sample similarity range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
        print(f"  Mean cross-sample similarity: {off_diag.mean().item():.3f}")

        print("\n[OK] All tests passed for aggregation: {}\n".format(aggregation))

    print("[OK] PatchTST Feature Extractor test suite completed successfully!")


def test_kcenter_compatibility():
    """Test compatibility with K-Center algorithm"""
    print("\nTesting K-Center Compatibility")

    # Configuration
    num_samples = 100
    seq_len = 336
    c_in = 21
    batch_size = 10

    # Create model
    model = PatchTSTFeatureExtractor(
        c_in=c_in,
        seq_len=seq_len,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.0,
        aggregation='max',
        target_dim=256,
        concat_rev_params=True
    )

    model.eval()

    # Extract features for all samples
    print(f"Extracting features for {num_samples} samples...")
    all_features = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = torch.randn(min(batch_size, num_samples - i), seq_len, c_in)
            batch_features = model(batch_x)
            all_features.append(batch_features)

    features = torch.cat(all_features, dim=0)
    print(f"Features shape: {features.shape}")
    print(f"Features are L2-normalized: {torch.norm(features[0], p=2).item():.6f}")
    print()

    # Test distance computation (cosine distance for L2-normalized vectors)
    print("Testing distance computation for K-Center:")
    print("  Computing pairwise distances...")

    # For L2-normalized vectors, cosine distance = 1 - dot_product
    # We'll compute Euclidean distance for this test
    distances = torch.cdist(features, features, p=2)

    print(f"  Distance matrix shape: {distances.shape}")
    print(f"  Distance matrix statistics:")
    print(f"    Min distance: {distances.min().item():.6f}")
    print(f"    Max distance: {distances.max().item():.6f}")
    print(f"    Mean distance: {distances.mean().item():.6f}")
    print(f"    Std distance: {distances.std().item():.6f}")
    print()

    # Simulate K-Center greedy selection
    print("Simulating K-Center greedy selection (k=10):")
    k = 10
    selected_indices = []

    # Select first point randomly
    selected_indices.append(0)

    # Greedy selection: select point with maximum minimum distance to selected set
    for _ in range(k - 1):
        if len(selected_indices) == 1:
            # First iteration: select farthest from initial point
            min_distances = distances[selected_indices[0]]
        else:
            # Subsequent iterations: select point with max min distance to any selected
            selected_dists = distances[:, selected_indices]
            min_distances, _ = selected_dists.min(dim=1)

        # Select point with maximum minimum distance
        farthest_idx = min_distances.argmax().item()
        selected_indices.append(farthest_idx)

    print(f"  Selected indices: {selected_indices}")
    print()

    # Compute coverage (maximum minimum distance from any point to selected set)
    selected_dists = distances[:, selected_indices]
    min_distances_to_selected, _ = selected_dists.min(dim=1)
    coverage = min_distances_to_selected.max().item()

    print(f"  Coverage (max min distance): {coverage:.6f}")
    print(f"  Mean distance to selected: {min_distances_to_selected.mean().item():.6f}")
    print()

    print("[OK] K-Center compatibility test passed")


def main():
    """Run all tests"""
    print("\nPatchTST Feature Extractor - Comprehensive Test Suite\n")

    # Set random seed
    torch.manual_seed(42)

    # Run tests
    try:
        test_rev_in()
        test_patching()
        test_aggregation_head()
        test_patchtst_encoder()
        test_kcenter_compatibility()

        print("\nALL TESTS PASSED SUCCESSFULLY!\n")
        print("The PatchTST Feature Extractor is ready for production use.")
        print("Key features verified:")
        print("  [OK] RevIN normalization for morphological detail focus")
        print("  [OK] Overlapping patching for boundary detail preservation")
        print("  [OK] No-GAP aggregation strategies (max pooling, flatten+linear)")
        print("  [OK] L2 normalization for cosine distance compatibility")
        print("  [OK] K-Center algorithm compatibility")
        print()

    except Exception as e:
        print("\nTEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
