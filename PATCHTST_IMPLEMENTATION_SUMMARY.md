# PatchTST Feature Extractor Implementation Summary

## Overview

Successfully implemented a modified PatchTST feature extractor for capturing microscopic details in multivariate time series data, specifically designed for use with K-Center greedy algorithm.

## Key Features Implemented

### 1. **Overlapping Micro-Patching**
- Small patch length (default: 16) for fine-grained analysis
- Stride < patch_len (default: 8, 50% overlap)
- Prevents information loss at patch boundaries
- Preserves local mutations and high-frequency glitches

### 2. **RevIN (Reversible Instance Normalization)**
- Removes macro-level mean and variance
- Forces model to learn pure morphological patterns
- Extracted parameters (mean, std) can be concatenated to output
- Makes representation scale-aware for K-Center

### 3. **No-GAP Aggregation Strategies**
- **Max Pooling**: Extracts most significant local activation features
- **Flatten + Linear**: Preserves temporal position of micro-features
- Both strategies avoid smoothing local anomalies (unlike Global Average Pooling)

### 4. **L2 Normalization**
- Final output normalized to unit length
- Ready for cosine distance computation in K-Center
- Ensures stable distance metrics

### 5. **Modular Architecture**
- `RevIN`: Normalization module
- `Patching`: Overlapping patch creation
- `TransformerEncoder`: Multi-layer self-attention
- `AggregationHead`: Configurable aggregation strategies
- `PatchTSTFeatureExtractor`: Complete feature extraction pipeline

## File Structure

```
src/layers/
  └── PatchTSTEncoder.py          # Main implementation

scripts/
  ├── pretrain_encoder.py          # Updated for PatchTST
  └── get_condensed_dataset.py     # Updated for PatchTST

run_full_pipeline.py                # Updated with PatchTST parameters
test_patchtst_encoder.py           # Comprehensive test suite
```

## Usage Examples

### 1. Test the Implementation
```bash
python test_patchtst_encoder.py
```

### 2. Pretrain the Encoder
```bash
python scripts/pretrain_encoder.py \\
    --data_path dataset/weather.csv \\
    --seq_len 336 \\
    --patch_len 16 \\
    --stride 8 \\
    --d_model 128 \\
    --epochs 30
```

### 3. Get Condensed Dataset
```bash
python scripts/get_condensed_dataset.py \\
    --algorithm kcenter \\
    --k 500 \\
    --patch_len 16 \\
    --stride 8 \\
    --aggregation max
```

### 4. Run Full Pipeline
```bash
python run_full_pipeline.py \\
    --data_path dataset/weather.csv \\
    --k 500 \\
    --patch_len 16 \\
    --stride 8 \\
    --aggregation max \\
    --pretrain_epochs 30
```

## Configuration Parameters

### PatchTST Specific
- `--patch_len`: Patch length (default: 16, recommended: 8-16)
- `--stride`: Stride between patches (default: 8, must be < patch_len)
- `--aggregation`: Aggregation strategy ('max' or 'flatten')
- `--target_dim`: Target output dimension for 'flatten' (default: 256)
- `--concat_rev_params`: Concatenate RevIN parameters (default: True)

### Standard Parameters
- `--seq_len`: Input sequence length (default: 336)
- `--c_in`: Input feature dimension (default: 21)
- `--d_model`: Transformer dimension (default: 128)
- `--n_heads`: Number of attention heads (default: 4)
- `--e_layers`: Number of encoder layers (default: 2)
- `--d_ff`: Feed-forward network dimension (default: 256)

## Design Principles

### Why This Architecture Captures Microscopic Details

1. **Overlapping Patches**
   - Small patches (16 timesteps) capture fine-grained patterns
   - 50% overlap ensures no boundary information loss
   - Each local feature is analyzed from multiple overlapping perspectives

2. **RevIN Normalization**
   - Removes absolute scale, focuses on relative patterns
   - Two time series with similar shapes but different scales map to similar features
   - Concatenating mean/std makes the representation scale-aware when needed

3. **No Global Average Pooling**
   - GAP would smooth out local anomalies (like a low-pass filter)
   - Max pooling preserves the most significant local activations
   - Flatten+Linear preserves temporal position information

4. **L2 Normalization**
   - Normalized vectors work well with cosine distance
   - K-Center greedy algorithm relies on distance metrics
   - Unit length ensures numerical stability

## Test Results

All tests passed successfully:
- ✓ RevIN normalization
- ✓ Overlapping patching
- ✓ Aggregation strategies (max pooling, flatten+linear)
- ✓ L2 normalization
- ✓ K-Center compatibility

Sample output:
- Output shape: (batch_size, 170) for max aggregation with RevIN concat
- L2 norms: 1.0 (perfectly normalized)
- Cross-sample similarity: 0.947 ± 0.006 (good diversity)

## Integration with Pipeline

The implementation is fully integrated with the existing pipeline:

1. **Pretraining**: `scripts/pretrain_encoder.py`
   - Trains PatchTST with masked reconstruction
   - Saves encoder-only weights for data selection

2. **Data Selection**: `scripts/get_condensed_dataset.py`
   - Loads pretrained PatchTST encoder
   - Extracts features for K-Center/Herding algorithms

3. **Training**: Uses condensed datasets for downstream tasks

4. **Pipeline**: `run_full_pipeline.py`
   - Orchestrates the entire process
   - Supports all PatchTST parameters

## Recommendations

### For Maximum Microscopic Detail Capture:
1. Use `patch_len=8` for very fine-grained analysis
2. Keep `stride=patch_len/2` for 50% overlap
3. Use `aggregation='flatten'` to preserve temporal position
4. Enable `concat_rev_params=True` for scale-aware features

### For Faster Processing:
1. Use `patch_len=16` (default)
2. Use `aggregation='max'` (fewer parameters)
3. Reduce `d_model` to 64 or 96 (if dimensionality is too high)

## Technical Details

### Architecture Diagram

```
Input (B, L, C)
    ↓
RevIN Normalization → (mean, std)
    ↓
Patching (overlap: stride < patch_len)
    ↓
Projection → (B, num_patches, d_model)
    ↓
Positional Encoding (learnable)
    ↓
Transformer Encoder (e_layers)
    ↓
Aggregation
    ├─ Max Pooling → (B, d_model)
    └─ Flatten+Linear → (B, target_dim)
    ↓
Concat RevIN params → (B, d_model + 2*C)
    ↓
L2 Normalization → (B, output_dim, ||v||₂ = 1)
```

### Output Dimensions

- Max aggregation with RevIN: `d_model + 2 * c_in`
- Flatten aggregation with RevIN: `target_dim + 2 * c_in`
- Without RevIN: `d_model` or `target_dim`

Example (default config):
- Max aggregation: 128 + 2*21 = 170 dimensions
- Flatten aggregation: 256 + 2*21 = 298 dimensions

## Future Enhancements

Possible improvements:
1. Add attention visualization for interpretability
2. Support for variable-length sequences
3. Multi-scale patching (multiple patch lengths)
4. Adaptive stride based on data characteristics
5. Integration with other selection algorithms beyond K-Center

## Troubleshooting

### Common Issues

1. **stride >= patch_len error**
   - Solution: Ensure `stride < patch_len` for overlapping patches

2. **CUDA out of memory**
   - Solution: Reduce `batch_size` or `d_model`

3. **Slow training**
   - Solution: Reduce `pretrain_epochs` or use smaller dataset

4. **Poor feature diversity**
   - Solution: Try different `aggregation` strategy or reduce `d_model`

## Citation

If you use this implementation, please cite:
- Original PatchTST paper
- This project's repository

## Contact

For issues or questions, please open an issue on the project repository.
