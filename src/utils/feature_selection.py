"""
Feature-Based Data Subset Selection for Time Series Forecasting.

基于 Transformer Encoder 高维特征的两种数据选择算法：
1. K-Center: 在高维特征空间中选择 K 个中心，使得所有点到最近中心的最大距离最小。
2. Herding: 在高维特征空间中迭代选择样本，使得每个新选中的样本距离已选中样本集的平均值最远。

使用流程:
    1. 使用 Transformer Encoder 将原始时序数据映射到高维特征空间
    2. 在特征空间中执行 K-Center 或 Herding 算法选择代表性样本
    3. 保存浓缩数据集用于后续训���
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from pathlib import Path
import pickle


# ============================================================================
# K-Center Algorithm (in Feature Space)
# ============================================================================

def kcenter(
    X: np.ndarray,
    k: int,
    use_gpu: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    在高维特征空间中执行 K-Center 选择（2-近似贪婪算法）。

    每次迭代选择距离当前所有中心最远的点作为新中心，直到选满 K 个。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵，shape (n_samples, feat_dim)。
    k : int
        选择的中心数量。
    use_gpu : bool
        是否使用 GPU 加速。
    seed : int
        随机种子。

    Returns
    -------
    selected_indices : np.ndarray
        选中的 K 个样本索引，shape (k,)。
    radius : float
        覆盖半径，即所有点到最近中心的最大距离。
    """
    print(f"[K-Center] Feature shape: {X.shape}, selecting k={k}")

    if use_gpu and torch.cuda.is_available():
        print("[K-Center] Using GPU acceleration")
        selected, radius = _kcenter_gpu(X, k, seed)
    else:
        print("[K-Center] Using CPU")
        selected, radius = _kcenter_cpu(X, k, seed)

    print(f"[K-Center] Selected {len(selected)} samples, radius={radius:.4f}")
    return selected, radius


def _kcenter_cpu(
    X: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """CPU 版 K-Center 实现。"""
    np.random.seed(seed)

    X = np.asarray(X, dtype=np.float64)
    n = len(X)

    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # 第一个中心：随机选择
    selected = [np.random.randint(0, n)]
    centers = X[selected[-1]].reshape(1, -1)

    # 迭代选剩余 k-1 个中心
    for _ in range(k - 1):
        # 计算每个点到所有已选中心的距离，取最小值
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        min_dists = dists.min(axis=1)
        # 选择距所有中心最远的点
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        centers = np.vstack([centers, X[idx]])

    # 计算最终覆盖半径
    final_dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    min_dists_final = final_dists.min(axis=1)
    radius = float(min_dists_final.max())

    return np.array(selected, dtype=int), radius


def _kcenter_gpu(
    X: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """GPU 加速版 K-Center 实现（chunked 计算避免显存溢出）。"""
    torch.manual_seed(seed)

    device = torch.device('cuda')
    X_tensor = torch.from_numpy(X).float().to(device)
    X_flat = X_tensor.reshape(X_tensor.shape[0], -1)
    n = X_flat.shape[0]

    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # 第一个中心：随机选择
    first_idx = torch.randint(0, n, (1,), device=device).item()
    selected = [first_idx]
    centers = X_flat[first_idx].clone().unsqueeze(0)

    # 分块大小，避免 n×k 距离矩阵过大
    BLOCK = 4096

    def _min_dist_to_centers(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算 x 中每点到 centers 的最近距离。"""
        min_dists = torch.full((x.shape[0],), float('inf'), device=device)
        for i in range(0, c.shape[0], BLOCK):
            chunk = c[i:i + BLOCK]
            dists = torch.cdist(x, chunk, p=2)
            chunk_min = dists.min(dim=1).values
            min_dists = torch.minimum(min_dists, chunk_min)
        return min_dists

    for _ in range(k - 1):
        min_dists = _min_dist_to_centers(X_flat, centers)
        idx = int(torch.argmax(min_dists).item())
        selected.append(idx)
        centers = torch.cat([centers, X_flat[idx].unsqueeze(0)], dim=0)

    # 最终半径
    final_min_dists = _min_dist_to_centers(X_flat, centers)
    radius = float(final_min_dists.max().item())

    torch.cuda.empty_cache()
    return np.array(selected, dtype=int), radius


# ============================================================================
# Herding Algorithm (in Feature Space)
# ============================================================================

def herding(
    X: np.ndarray,
    k: int,
    use_gpu: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    在高维特征空间中执行 Herding 选择。

    每次迭代选择距离已选中样本集的平均值（中心）最远的点。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵，shape (n_samples, feat_dim)。
    k : int
        选择的样本数量。
    use_gpu : bool
        是否使用 GPU 加速。
    seed : int
        随机种子。

    Returns
    -------
    selected_indices : np.ndarray
        选中的 K 个样本索引，shape (k,)。
    radius : float
        最终覆盖半径。
    """
    print(f"[Herding] Feature shape: {X.shape}, selecting k={k}")

    if use_gpu and torch.cuda.is_available():
        print("[Herding] Using GPU acceleration")
        selected, radius = _herding_gpu(X, k, seed)
    else:
        print("[Herding] Using CPU")
        selected, radius = _herding_cpu(X, k, seed)

    print(f"[Herding] Selected {len(selected)} samples, radius={radius:.4f}")
    return selected, radius


def _herding_cpu(
    X: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """CPU 版 Herding 实现。"""
    np.random.seed(seed)

    X = np.asarray(X, dtype=np.float64)
    n = len(X)

    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # 第一个样本：随机选择
    selected = [np.random.randint(0, n)]
    selected_set = set(selected)

    # 迭代选剩余 k-1 个样本
    for _ in range(k - 1):
        # 计算已选中样本的平均（中心）
        center = X[selected].mean(axis=0, keepdims=True)

        # 计算每个点到平均中心的距离
        dists = np.linalg.norm(X - center, axis=1)

        # 排除已选中的样本
        dists[list(selected_set)] = -np.inf

        # 选择距离平均中心最远的点
        idx = int(np.argmax(dists))
        selected.append(idx)
        selected_set.add(idx)

    # 计算最终覆盖半径
    final_center = X[selected].mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(X - final_center, axis=1).max())

    return np.array(selected, dtype=int), radius


def _herding_gpu(
    X: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """GPU 加速版 Herding 实现。"""
    torch.manual_seed(seed)

    device = torch.device('cuda')
    X_tensor = torch.from_numpy(X).float().to(device)
    X_flat = X_tensor.reshape(X_tensor.shape[0], -1)
    n = X_flat.shape[0]

    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # 第一个样本：随机选择
    first_idx = torch.randint(0, n, (1,), device=device).item()
    selected = [first_idx]
    selected_set = set(selected)

    for _ in range(k - 1):
        # 计算已选中样本的平均（中心）
        selected_tensor = torch.tensor(selected, device=device)
        center = X_flat[selected_tensor].mean(dim=0, keepdim=True)

        # 计算每个点到平均中心的距离
        dists = torch.norm(X_flat - center, p=2, dim=1)

        # 排除已选中的样本
        dists[list(selected_set)] = -float('inf')

        # 选择距离平均中心最远的点
        idx = int(torch.argmax(dists).item())
        selected.append(idx)
        selected_set.add(idx)

    # 最终半径
    selected_tensor = torch.tensor(selected, device=device)
    final_center = X_flat[selected_tensor].mean(dim=0, keepdim=True)
    radius = float(torch.norm(X_flat - final_center, p=2, dim=1).max().item())

    torch.cuda.empty_cache()
    return np.array(selected, dtype=int), radius


# ============================================================================
# Feature Extraction with Transformer Encoder
# ============================================================================

def extract_features(
    data_loader: DataLoader,
    encoder: nn.Module,
    device: str = 'cuda'
) -> np.ndarray:
    """
    使用 Transformer Encoder 提取数据集的高维特征。

    Parameters
    ----------
    data_loader : DataLoader
        数据加载器
    encoder : nn.Module
        特征编码器（Transformer Encoder）
    device : str
        设备 ('cuda' or 'cpu')

    Returns
    -------
    features : np.ndarray
        高维特征，shape (n_samples, feat_dim)
    """
    encoder.eval()
    all_features = []

    print(f"[Extract] Extracting features from {len(data_loader.dataset)} samples...")

    with torch.no_grad():
        for batch_x, _, batch_x_mark, _ in data_loader:
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            # 提取特征 - PatchTST only needs x, old encoders may need x_mark
            try:
                features = encoder(batch_x, batch_x_mark)
            except TypeError:
                # PatchTST encoder only takes x as input
                features = encoder(batch_x)
            all_features.append(features.cpu().numpy())

    features = np.vstack(all_features)
    print(f"[Extract] Extracted features: {features.shape}")

    return features


# ============================================================================
# High-Level Selection Functions
# ============================================================================

def select_samples(
    data_loader: DataLoader,
    encoder: nn.Module,
    k: int,
    algorithm: str = 'kcenter',
    device: str = 'cuda'
) -> Tuple[np.ndarray, float]:
    """
    使用 Transformer Encoder 特征执行数据选择。

    流程:
        1. 使用 Encoder 将数据映射到高维特征空间
        2. 在特征空间中执行 K-Center 或 Herding

    Parameters
    ----------
    data_loader : DataLoader
        数据加载器
    encoder : nn.Module
        特征编码器
    k : int
        选择的���本数量
    algorithm : str
        选择算法: 'kcenter' 或 'herding'
    device : str
        设备

    Returns
    -------
    selected_indices : np.ndarray
        选中的样本索引
    radius : float
        覆盖半径
    """
    print("=" * 60)
    print(f"Feature-Based Selection: {algorithm.upper()}")
    print("=" * 60)

    # 提取特征
    features = extract_features(data_loader, encoder, device)
    print()

    # 执行选择
    if algorithm == 'kcenter':
        selected, radius = kcenter(features, k, use_gpu=(device == 'cuda'))
    elif algorithm == 'herding':
        selected, radius = herding(features, k, use_gpu=(device == 'cuda'))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return selected, radius


# ============================================================================
# Save and Load Condensed Dataset
# ============================================================================

def save_dataset(
    data_loader: DataLoader,
    selected_indices: np.ndarray,
    save_dir: str,
    metadata: Optional[dict] = None
) -> str:
    """
    保存浓缩数据集到磁盘。

    保存内容:
    - data_x.npy: 输入序列 (n_selected, seq_len, n_features)
    - data_y.npy: 标签序列 (n_selected, label_len+pred_len, n_features)
    - data_x_mark.npy: 输入时间标记 (n_selected, seq_len, time_features)
    - data_y_mark.npy: 标签时间标记 (n_selected, label_len+pred_len, time_features)
    - indices.npy: 选中的索引 (n_selected,)
    - metadata.pkl: 元数据

    Parameters
    ----------
    data_loader : DataLoader
        原始数据的 DataLoader
    selected_indices : np.ndarray
        选中的样本索引
    save_dir : str
        保存目录路径
    metadata : dict, optional
        额外的元数据

    Returns
    -------
    save_path : str
        保存目录的完整路径
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = data_loader.dataset

    print(f"[Save] Saving condensed dataset to {save_path}...")

    # 收集所有样本
    data_x_list = []
    data_y_list = []
    data_x_mark_list = []
    data_y_mark_list = []

    for idx in selected_indices:
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[idx]
        if hasattr(seq_x, 'numpy'):
            seq_x = seq_x.numpy()
            seq_y = seq_y.numpy()
            seq_x_mark = seq_x_mark.numpy()
            seq_y_mark = seq_y_mark.numpy()
        data_x_list.append(seq_x)
        data_y_list.append(seq_y)
        data_x_mark_list.append(seq_x_mark)
        data_y_mark_list.append(seq_y_mark)

    # 转换为 numpy 数组
    data_x = np.stack(data_x_list, axis=0)
    data_y = np.stack(data_y_list, axis=0)
    data_x_mark = np.stack(data_x_mark_list, axis=0)
    data_y_mark = np.stack(data_y_mark_list, axis=0)

    # 保存数据
    np.save(save_path / 'data_x.npy', data_x)
    np.save(save_path / 'data_y.npy', data_y)
    np.save(save_path / 'data_x_mark.npy', data_x_mark)
    np.save(save_path / 'data_y_mark.npy', data_y_mark)
    np.save(save_path / 'indices.npy', selected_indices)

    # 保存元数据
    meta = {
        'n_samples': len(selected_indices),
        'data_x_shape': data_x.shape,
        'data_y_shape': data_y.shape,
        'data_x_mark_shape': data_x_mark.shape,
        'data_y_mark_shape': data_y_mark.shape,
    }
    if metadata:
        meta.update(metadata)

    with open(save_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f"[Save] Saved {len(selected_indices)} samples to {save_path}")
    print(f"  - data_x: {data_x.shape}")
    print(f"  - data_y: {data_y.shape}")
    print(f"  - data_x_mark: {data_x_mark.shape}")
    print(f"  - data_y_mark: {data_y_mark.shape}")

    return str(save_path)


def load_dataset(load_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    从磁盘加载浓缩数据集。

    Parameters
    ----------
    load_dir : str
        数据集保存目录

    Returns
    -------
    data_x : np.ndarray
        输入序列
    data_y : np.ndarray
        标签序列
    data_x_mark : np.ndarray
        输入时间标记
    data_y_mark : np.ndarray
        标签时间标记
    metadata : dict
        元数据字典
    """
    load_path = Path(load_dir)

    print(f"[Load] Loading condensed dataset from {load_path}...")

    data_x = np.load(load_path / 'data_x.npy')
    data_y = np.load(load_path / 'data_y.npy')
    data_x_mark = np.load(load_path / 'data_x_mark.npy')
    data_y_mark = np.load(load_path / 'data_y_mark.npy')

    with open(load_path / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"[Load] Loaded {metadata['n_samples']} samples")
    print(f"  - data_x: {data_x.shape}")
    print(f"  - data_y: {data_y.shape}")
    print(f"  - data_x_mark: {data_x_mark.shape}")
    print(f"  - data_y_mark: {data_y_mark.shape}")

    return data_x, data_y, data_x_mark, data_y_mark, metadata


class CondensedDataset(nn.Module):
    """
    浓缩数据集类，从保存的文件加载数据并作为 PyTorch Dataset 使用。

    使用示例:
        dataset = CondensedDataset('condensed_datasets/weather_kcenter_k500')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, load_dir: str):
        """
        Parameters
        ----------
        load_dir : str
            浓缩数据集保存目录
        """
        load_path = Path(load_dir)

        # 加载数据
        self.data_x = np.load(load_path / 'data_x.npy')
        self.data_y = np.load(load_path / 'data_y.npy')
        self.data_x_mark = np.load(load_path / 'data_x_mark.npy')
        self.data_y_mark = np.load(load_path / 'data_y_mark.npy')

        # 加载元数据
        with open(load_path / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

        self._len = len(self.data_x)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data_x[idx], dtype=torch.float32),
            torch.tensor(self.data_y[idx], dtype=torch.float32),
            torch.tensor(self.data_x_mark[idx], dtype=torch.float32),
            torch.tensor(self.data_y_mark[idx], dtype=torch.float32),
        )
