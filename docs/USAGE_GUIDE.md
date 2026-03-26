# 浓缩数据集使用指南

本指南说明如何使用浓缩数据集训练时序预测模型。

## 目录

1. [预训练 Encoder（可选）](#预训练-encoder可选)
2. [生成浓缩数据集](#生成浓缩数据集)
3. [训练模型](#训练模型)
4. [完整示例](#完整示例)

---

## 预训练 Encoder（可选）

在生成浓缩数据集之前，可以先预训练 Transformer Encoder 以获得更好的特征表示。

### 预训练方法

使用 Masked Reconstruction 方法：
- 随机 mask 75% 的时间步
- 使用 Encoder 提取特征
- 通过 Decoder 重建被 mask 的时间步
- 最小化重建误差

### 基本用法

```bash
# 使用默认配置预训练（50 epochs）
python pretrain_encoder.py

# 自定义配置
python pretrain_encoder.py \
    --epochs 50 \
    --batch_size 32 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 2 \
    --mask_ratio 0.75 \
    --learning_rate 0.001 \
    --save_path checkpoints/encoder_pretrained.pth
```

### 预训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 50 |
| `--batch_size` | 批量大小 | 32 |
| `--d_model` | 模型维度 | 256 |
| `--n_heads` | 注意力头数 | 8 |
| `--e_layers` | 编码器层数 | 2 |
| `--d_ff` | FFN 维度 | 512 |
| `--mask_ratio` | Mask 比例 | 0.75 |
| `--learning_rate` | 学习率 | 0.001 |

### 预训练输出

预训练完成后会生成两个文件：
- `checkpoints/encoder_pretrained.pth` - 完整模型（包含Decoder）
- `checkpoints/encoder_pretrained_encoder_only.pth` - 仅Encoder

---

## 生成浓缩数据集

使用 `get_condensed_dataset.py` 从原始数据集中选择代表性样本。

### 步骤 1（可选）: 预训练 Encoder

```bash
# 使用默认配置预训练
python pretrain_encoder.py --epochs 50 --save_path checkpoints/my_encoder.pth
```

### 步骤 2: 生成浓缩数据集

#### 不使用预训练 Encoder

```bash
# K-Center 算法
python get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --d_model 128 \
    --save_dir condensed_datasets/weather_kcenter_k500

# Herding 算法
python get_condensed_dataset.py \
    --algorithm herding \
    --k 500 \
    --d_model 128 \
    --save_dir condensed_datasets/weather_herding_k500
```

#### 使用预训练 Encoder

```bash
# 使用预训练的 Encoder
python get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --d_model 128 \
    --pretrained_path checkpoints/my_encoder_encoder_only.pth \
    --save_dir condensed_datasets/weather_kcenter_k500_pretrained
```

#### 使用 Herding 算法（不使用预训练）

```bash
python get_condensed_dataset.py \
    --algorithm herding \
    --k 500 \
    --d_model 128 \
    --save_dir condensed_datasets/weather_herding_k500
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithm` | 选择算法: `kcenter` �� `herding` | `kcenter` |
| `--k` | 选择的样本数量 | 500 |
| `--data_path` | 原始数据集路径 | `dataset/weather.csv` |
| `--seq_len` | 输入序列长度 | 96 |
| `--pred_len` | 预测长度 | 96 |
| `--d_model` | 特征维度 | 128 |
| `--n_heads` | 注意力头数 | 4 |
| `--e_layers` | 编码器层数 | 2 |
| `--d_ff` | FFN 维度 | 256 |
| `--save_dir` | 保存目录 | 自动生成 |
| `--device` | 设备: `cuda` 或 `cpu` | `cuda` |

### 步骤 2: 使用浓缩数据集训练模型

使用 `train_with_condensed.py` 训练模型。

#### 训练 Autoformer

```bash
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --model Autoformer \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001
```

#### 训练 Transformer

```bash
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --model Transformer \
    --epochs 10
```

#### 训练 Informer

```bash
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --model Informer \
    --epochs 10
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--condensed_dir` | 浓缩数据集目录（必需） | - |
| `--model` | 模型类型 | `Autoformer` |
| `--epochs` | 训练轮数 | 10 |
| `--batch_size` | 批量大小 | 32 |
| `--learning_rate` | 学习率 | 0.0001 |
| `--d_model` | 模型维度 | 512 |
| `--n_heads` | 注意力头数 | 8 |
| `--patience` | Early stopping 耐心值 | 3 |

## 完整示例

### 示例 1: K-Center + Autoformer

```bash
# 1. 生成浓缩数据集 (K-Center, 500 个样本)
python get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --d_model 128 \
    --n_heads 4

# 2. 训练 Autoformer
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --model Autoformer \
    --epochs 10 \
    --batch_size 32
```

### 示例 2: Herding + Transformer

```bash
# 1. 生成浓缩数据集 (Herding, 1000 个样本)
python get_condensed_dataset.py \
    --algorithm herding \
    --k 1000 \
    --d_model 256

# 2. 训练 Transformer
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_herding_k1000 \
    --model Transformer \
    --epochs 20
```

### 示例 3: 完整流程（自定义参数）

```bash
# 1. 生成浓缩数据集
python get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --save_dir my_condensed_dataset

# 2. 训练模型
python train_with_condensed.py \
    --condensed_dir my_condensed_dataset \
    --model Autoformer \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --patience 3
```

## 程序化使用

### Python API 使用

```python
from feature_selection import select_samples, save_dataset, CondensedDataset
from layers.FeatureEncoder import create_feature_encoder
from torch.utils.data import DataLoader
from process.data_loader import Dataset_Custom
from models import Autoformer

# 1. 加载数据集
train_set = Dataset_Custom(
    root_path='./',
    data_path='dataset/weather.csv',
    flag='train',
    size=[96, 96, 96],
    features='M',
    target='OT',
    scale=True,
    timeenc=0,
    freq='10min'
)

train_loader = DataLoader(train_set, batch_size=256, shuffle=False)

# 2. 创建编码器
encoder = create_feature_encoder(
    c_in=21,
    d_model=128,
    n_heads=4,
    e_layers=2,
    d_ff=256,
    device='cuda'
)

# 3. 执行数据选择
selected_indices, radius = select_samples(
    train_loader,
    encoder,
    k=500,
    algorithm='kcenter',
    device='cuda'
)

# 4. 保存浓缩数据集
save_dataset(
    train_loader,
    selected_indices,
    save_dir='condensed_datasets/my_kcenter_k500',
    metadata={'algorithm': 'kcenter', 'd_model': 128, 'radius': radius}
)

# 5. 加载浓缩数据集并训练
condensed_dataset = CondensedDataset('condensed_datasets/my_kcenter_k500')
condensed_loader = DataLoader(condensed_dataset, batch_size=32, shuffle=True)

# 创建模型并训练
model = Autoformer(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    for batch_x, batch_y, batch_x_mark, batch_y_mark in condensed_loader:
        # 训练代码...
        pass
```

## 文件结构

生成的浓缩数据集目录结构：

```
condensed_datasets/weather_kcenter_k500/
├── data_x.npy           # 输入序列 (500, 96, 21)
├── data_y.npy           # 标签序列 (500, 192, 21)
├── data_x_mark.npy      # 输入时间标记 (500, 96, 5)
├── data_y_mark.npy      # 标签时间标记 (500, 192, 5)
├── indices.npy          # 选中的样本索引 (500,)
└── metadata.pkl         # 元数据
```

## 元数据说明

保存的 `metadata.pkl` 包含：

- `algorithm`: 选择算法（kcenter/herding）
- `method`: 'feature_based'
- `d_model`: 特征维度
- `radius`: 覆盖半径
- `k`: 选择的样本数
- `seq_len`: 序列长度
- `pred_len`: 预测长度
- `original_size`: 原始数据集大小
- `condensed_size`: 浓缩数据集大小
- `compression_ratio`: 压缩比例

## 性能对比建议

对比不同算法和不同 K 值的效果：

```bash
# 测试不同 K 值
for k in 200 500 1000 2000; do
    python get_condensed_dataset.py --algorithm kcenter --k $k --save_dir condensed_datasets/kcenter_k${k}
    python train_with_condensed.py --condensed_dir condensed_datasets/kcenter_k${k} --epochs 10
done

# 测试不同算法
for algo in kcenter herding; do
    python get_condensed_dataset.py --algorithm ${algo} --k 500 --save_dir condensed_datasets/${algo}_k500
    python train_with_condensed.py --condensed_dir condensed_datasets/${algo}_k500 --epochs 10
done
```

## 常见问题

### 1. 内存不足

减小批量大小或特征维度：

```bash
python train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --batch_size 16 \
    --d_model 256
```

### 2. GPU 不可用

使用 CPU：

```bash
python get_condensed_dataset.py --algorithm kcenter --k 500 --device cpu
python train_with_condensed.py --condensed_dir ... --use_gpu False
```

### 3. 查看浓缩数据集信息

```python
from feature_selection import CondensedDataset

dataset = CondensedDataset('condensed_datasets/weather_kcenter_k500')
print(f"数据集大小: {len(dataset)}")
print(f"元数据: {dataset.metadata}")
```

## 更多帮助

- 查看 [README.md](README.md) 了解完整项目信息
- 查看 [QUICKSTART.md](QUICKSTART.md) 了解快速参考
- 查看 [examples/](examples/) 目录查看更多示例代码
