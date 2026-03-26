# CondensedTSF: Data Condensation for Time Series Forecasting

基于数据浓缩算法（K-Center & Herding）的高效时序预测框架。

## 项目特点

**CondensedTSF** 旨在通过智能数据选择技术提升时序预测训练效率：

- 🎯 **数据浓缩算法**：支持 K-Center 和 Herding 两种数据选择方法
- 🤖 **多模型支持**：集成 Autoformer、Transformer、Informer、Reformer 等模型
- 🚀 **高效训练**：通过浓缩数据集大幅减少训练时间，保持预测精度
- 🔧 **灵活配置**：支持预训练编码器、自定义浓缩比例等丰富配置

## 项目结构

```
CondensedTSF/
├── README.md              # 项目说明
├── requirements.txt       # 依赖包列表
├── setup.py              # 安装配置
│
├── src/                  # 源代码
│   ├── __init__.py
│   ├── models/          # 模型定义
│   │   ├── Autoformer.py
│   │   ├── Transformer.py
│   │   ├── Informer.py
│   │   └── Reformer.py
│   ├── layers/          # 神经网络层
│   │   ├── AutoCorrelation.py
│   │   ├── Autoformer_EncDec.py
│   │   ├── Embed.py
│   │   ├── SelfAttention_Family.py
│   │   ├── Transformer_EncDec.py
│   │   ├── masking.py
│   │   └── FeatureEncoder.py
│   ├── data/            # 数据处理
│   │   ├── data_loader.py
│   │   └── timefeatures.py
│   ├── utils/           # 工具函数
│   │   ├── metrics.py
│   │   ├── feature_selection.py
│   │   └── __init__.py
│   └── exp/             # 实验管理
│       ├── exp_basic.py
│       └── exp_main.py
│
├── scripts/             # 可执行脚本
│   ├── train.py                    # 训练模型
│   ├── test_model.py              # 测试模型
│   ├── pretrain_encoder.py        # 预训练编码器
│   ├── train_with_condensed.py    # 使用浓缩数据集训练
│   └── get_condensed_dataset.py   # 获取浓缩数据集
│
├── configs/             # 配置文件
│   └── config.py
│
├── docs/                # 文档
│   ├── STRUCTURE.md
│   ├── USAGE_GUIDE.md
│   └── QUICKSTART.md
│
├── tests/               # 测试文件
│
├── dataset/             # 数据集
│   └── weather.csv
│
└── outputs/             # 输出目录
    ├── checkpoints/     # 模型检查点
    ├── condensed_datasets/  # 浓缩数据集
    ├── logs/           # 训练日志
    └── results/        # 实验结果
```

## 安装

### 基础安装

```bash
# 克隆项目
git clone <repository-url>
cd CondensedTSF

# 安装依赖
pip install -r requirements.txt
```

### 依赖包

主要依赖：
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm

（可选）Reformer 模型需要额外安装：
```bash
pip install reformer-pytorch
```

## 快速开始

### 1. 基础测试

测试所有模型的前向传播：

```bash
# 测试 Autoformer（默认）
python scripts/test_model.py --model Autoformer

# 测试 Transformer
python scripts/test_model.py --model Transformer

# 自定义参数
python scripts/test_model.py --model Autoformer --seq_len 96 --pred_len 96 --d_model 512
```

### 2. 训练模型

```bash
# 使用默认参数训练 Autoformer
python scripts/train.py --model Autoformer --is_training 1

# 训练 Transformer，10 个 epoch
python scripts/train.py --model Transformer --is_training 1 --train_epochs 10

# 训练 Informer，自定义参数
python scripts/train.py --model Informer --is_training 1 \
    --seq_len 96 --pred_len 96 \
    --d_model 512 --n_heads 8 \
    --batch_size 32 --learning_rate 0.0001
```

### 3. 预训练编码器

```bash
# 预训练 Transformer Encoder
python scripts/pretrain_encoder.py \
    --epochs 50 \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --save_path outputs/checkpoints/encoder_pretrained.pth
```

### 4. 获取浓缩数据集

使用 K-Center 或 Herding 算法生成浓缩数据集：

```bash
# 使用 K-Center 算法
python scripts/get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --save_dir outputs/condensed_datasets/weather_kcenter_k500

# 使用 Herding 算法
python scripts/get_condensed_dataset.py \
    --algorithm herding \
    --k 1000

# 使用预训练编码器
python scripts/get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --pretrained_path outputs/checkpoints/encoder_pretrained.pth
```

### 5. 使用浓缩数据集训练

```bash
python scripts/train_with_condensed.py \
    --condensed_dir outputs/condensed_datasets/weather_kcenter_k500 \
    --model Autoformer \
    --epochs 10 \
    --batch_size 32
```

### 6. 测试已训练模型

```bash
# 加载 checkpoint 进行测试
python scripts/train.py --model Autoformer --is_training 0
```

## 配置文件

使用预定义配置简化参数设置：

```python
from configs.config import get_config

# 获取配置
config = get_config('weather')

# 使用配置参数
print(f"模型: {config.model}")
print(f"序列长度: {config.seq_len}")
```

## 数据集格式

数据集采用 CSV 格式，第一列为日期（`date`），其余列为特征。

示例（`dataset/weather.csv`）：

```csv
date,Humidity,Light,Pressure,Temperature,...,OT
2020-01-01 00:00:00,86.2,79.2,1009.0,16.3,...,16.3
2020-01-01 00:10:00,85.8,79.1,1009.1,16.5,...,16.5
...
```

- **date**: 时间戳（必填）
- **OT**: 目标预测列（通过 `--target` 参数指定）
- 其他列：输入特征

## 核心功能

### 1. 模型支持

- **Autoformer**: 自相关注意力 + 序列分解
- **Transformer**: 标准自注意力机制
- **Informer**: ProbSparse 注意力机制
- **Reformer**: LSH 注意力机制

### 2. 数据浓缩算法

- **K-Center**: 基于特征空间的贪心算法，最小化最大覆盖半径
- **Herding**: 类似 K-Means 的中心选择算法，保持数据分布特性
- **特征空间映射**: 使用 Transformer Encoder 将时序数据映射到高维语义空间
- **预训练支持**: 支持 Masked Reconstruction 预训练方法提取更优特征

### 3. 预训练

使用 Masked Reconstruction 方法预训练特征编码器：
- 随机 mask 部分时间步
- 使用 Encoder 提取特征
- 通过 Decoder 重建被 mask 的时间步
- 最小化重建误差

## 参数说明

### 模型选择

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--model` | 模型名称 | `Autoformer`, `Transformer`, `Informer`, `Reformer` |

### 数据参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 数据集路径 | `dataset/weather.csv` |
| `--features` | 预测任务 | `M` (多变量→多变量) |
| `--target` | 目标列 | `OT` |
| `--freq` | 时间频率 | `10min`, `h`, `d`, `t` 等 |
| `--seq_len` | 输入序列长度 | 96 |
| `--label_len` | 解码器起始长度 | 96 |
| `--pred_len` | 预测长度 | 96 |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enc_in` | 编码器输入维度 | 21 (特征数) |
| `--dec_in` | 解码器输入维度 | 21 |
| `--c_out` | 输出维度 | 21 |
| `--d_model` | 模型维度 | 512 |
| `--n_heads` | 注意力头数 | 8 |
| `--e_layers` | 编码器层数 | 2 |
| `--d_layers` | 解码器层数 | 1 |
| `--d_ff` | 前馈网络维度 | 2048 |
| `--moving_avg` | 移动平均窗口 | 25 |
| `--factor` | 注意力因子 | 1 |
| `--dropout` | Dropout 率 | 0.05 |
| `--activation` | 激活函数 | `gelu` |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--is_training` | 是否训练 | 1 (训练) / 0 (测试) |
| `--train_epochs` | 训练轮数 | 10 |
| `--batch_size` | 批量大小 | 32 |
| `--learning_rate` | 学习率 | 0.0001 |
| `--patience` | Early stopping 耐心值 | 3 |
| `--use_gpu` | 使用 GPU | True |
| `--gpu` | GPU 设备 ID | 0 |

## 输出说明

训练后生成的文件：

```
outputs/
├── checkpoints/
│   └── weather_Autoformer_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/
│       └── checkpoint.pth
│
├── condensed_datasets/
│   └── weather_kcenter_k500/
│       ├── condensed_data.npz
│       ├── selected_indices.npy
│       └── metadata.json
│
├── logs/
│   └── training_*.log
│
└── results/
    └── weather_Autoformer_ftM_sl96_ll96_pl96_..._test_0/
        ├── metrics.npy
        ├── pred.npy
        └── true.npy
```

## 模型对比

| 模型 | 复杂度 | 特点 | 适用场景 |
|------|--------|------|----------|
| **Autoformer** | O(L log L) | 自相关注意力 + 序列分解 | 长序列预测，周期性数据 |
| **Transformer** | O(L²) | 标准自注意力 | 中短序列，复杂模式 |
| **Informer** | O(L log L) | ProbSparse 注意力 | 长序列预测 |
| **Reformer** | O(L log L) | LSH 注意力 | 超长序列 |

## 常见问题

### 1. GPU 内存不足

减小批量大小或模型维度：

```bash
python scripts/train.py --batch_size 16 --d_model 256
```

### 2. 数据集时间特征维度错误

确保 `freq` 参数与数据集匹配：
- `10min`: 10 分钟数据 → 5 维时间特征 [月, 日, 星期, 小时, 分钟]
- `h`: 小时数据 → 4 维 [月, 日, 星期, 小时]
- `d`: 日数据 → 3 维 [月, 日, 星期]

### 3. Reformer 模型报错

安装额外依赖：

```bash
pip install reformer-pytorch
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 引用

如果使用本项目，请引用原始论文：

```bibtex
@inproceedings{autoformer,
  title={Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
  author={Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICLR},
  year={2022}
}
```

## License

MIT License
