# K-Center 项目结构说明

## 📁 完整项目结构

```
K-Center/
├── README.md                   # 📖 项目说明文档
├── QUICKSTART.md               # 🚀 快速参考指南
├── requirements.txt            # 📦 依赖包列表
├── config.py                   # ⚙️  配置文件
│
├── 🎯 主程序
├── run.py                      # 训练/测试入口脚本
├── test.py                     # 快速测试脚本
├── kcenter.py                  # K-Center 选择算法
│
├── 📊 数据处理 (process/)
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载器 (Dataset_Custom)
│   └── timefeatures.py         # 时间特征提取
│
├── 🧠 神经网络层 (layers/)
│   ├── __init__.py
│   ├── AutoCorrelation.py      # 自相关注意力机制
│   ├── Autoformer_EncDec.py    # Autoformer 编解码器
│   ├── Embed.py                # 嵌入层 (Token/Position/Temporal)
│   ├── SelfAttention_Family.py # 注意力机制族
│   ├── Transformer_EncDec.py   # Transformer 编解码器
│   └── masking.py              # 注意力掩码
│
├── 🤖 模型定义 (models/)
│   ├── __init__.py
│   ├── Autoformer.py           # Autoformer 模型
│   ├── Transformer.py          # Transformer 模型
│   ├── Informer.py             # Informer 模型
│   └── Reformer.py             # Reformer 模型
│
├── 🔬 实验管理 (exp/)
│   ├── __init__.py
│   ├── exp_basic.py            # 实验基类
│   └── exp_main.py             # 主实验类 (训练/测试)
│
├── 🛠️ 工具函数 (utils/)
│   ├── __init__.py
│   └── metrics.py              # 评估指标 (MAE, MSE, RMSE, MAPE, MSPE)
│
├── 📈 数据采样方法 (data/)
│   └── generating.py           # K-Means, Herding, Gradient Matching
│
├── 📚 示例代码 (examples/)
│   ├── 01_basic_test.py        # 基础模型测试
│   ├── 02_kcenter_selection.py # K-Center 数据选择
│   └── 03_train_with_kcenter.py # 使用 K-Center 子集训练
│
├── 💾 数据集 (dataset/)
│   └── weather.csv             # 天气数据集 (21 特征)
│
└── 📁 输出目录
    ├── checkpoints/            # 模型检查点
    ├── results/                # 实验结果 (metrics, predictions)
    └── test_results/           # 测试可视化图表
```

## 🔧 模块说明

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **数据加载** | `process/data_loader.py` | 提供 `Dataset_Custom` 类，加载 CSV 格式的时序数据 |
| **K-Center** | `kcenter.py` | 提供贪婪 K-Center 算法，支持 GPU 加速 |
| **训练框架** | `exp/exp_main.py` | 封装训练、验证、测试流程 |

### 模型层 (layers/)

| 文件 | 类 | 功能 |
|------|-----|------|
| `AutoCorrelation.py` | `AutoCorrelation` | 基于自相关的注意力机制 |
| `Autoformer_EncDec.py` | `Encoder`, `Decoder` | Autoformer 的编解码器 |
| `Embed.py` | `DataEmbedding` | Token + Position + Temporal 嵌入 |
| `SelfAttention_Family.py` | `FullAttention`, `ProbAttention` | 注意力机制实现 |
| `Transformer_EncDec.py` | `Encoder`, `Decoder` | 标准 Transformer 编解码器 |

### 模型 (models/)

| 模型 | 复杂度 | 特点 |
|------|--------|------|
| `Autoformer` | O(L log L) | 自相关注意力 + 序列分解 |
| `Transformer` | O(L²) | 标准自注意力 |
| `Informer` | O(L log L) | ProbSparse 注意力 |
| `Reformer` | O(L log L) | LSH 注意力 (需 `reformer-pytorch`) |

## 📝 数据流

```
原始数据 (CSV)
    ↓
process/data_loader.py
    ├─ StandardScaler 标准化
    ├─ 时间特征提取 [月, 日, 星期, 小时, 分钟]
    └─ 切分为 train/val/test
    ↓
torch.utils.data.DataLoader
    ├─ 批量加载
    └─ 数据增强 (可选)
    ↓
[可选] K-Center 选择
    ├─ 选择 K 个代表性样本
    └─ 创建子集 DataLoader
    ↓
模型训练 (exp/exp_main.py)
    ├─ 前向传播
    ├─ 损失计算 (MSE)
    ├─ 反向传播
    └─ Early Stopping
    ↓
模型评估
    ├─ MSE, MAE, RMSE
    └─ 可视化图表
```

## 🎯 典型工作流程

### 1. 基础测试

```bash
# 测试所有模型是否正常工作
python examples/01_basic_test.py
```

### 2. 使用 K-Center 选择数据

```bash
# 从完整训练集中选择 500 个样本
python examples/02_kcenter_selection.py
```

### 3. 训练模型

```bash
# 方法 A: 使用 run.py
python run.py --model Autoformer --is_training 1 --train_epochs 10

# 方法 B: 使用示例代码
python examples/03_train_with_kcenter.py
```

### 4. 评估模型

```bash
# 加载 checkpoint 并测试
python run.py --model Autoformer --is_training 0
```

## 📊 输入输出格式

### 输入 (CSV)

```csv
date, feature1, feature2, ..., target
2020-01-01 00:00:00, 0.5, 0.3, ..., 0.8
2020-01-01 00:10:00, 0.6, 0.4, ..., 0.9
...
```

### 输出 (PyTorch Tensor)

| 变量 | Shape | 说明 |
|------|-------|------|
| `batch_x` | `[B, seq_len, C]` | 输入序列 |
| `batch_y` | `[B, label_len+pred_len, C]` | 标签序列 |
| `batch_x_mark` | `[B, seq_len, T]` | 输入时间特征 |
| `batch_y_mark` | `[B, label_len+pred_len, T]` | 标签时间特征 |
| `output` | `[B, pred_len, C]` | 预测结果 |

其中：
- `B` = batch size
- `C` = 特征数 (如 weather.csv 为 21)
- `T` = 时间特征维度 (10min 数据为 5, 小时数据为 4)

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 1. 测试模型
python examples/01_basic_test.py

# 2. K-Center 选择
python examples/02_kcenter_selection.py

# 3. 训练模型
python run.py --model Autoformer --is_training 1
```

### 自定义配置

```python
# config.py
from config import get_config

config = get_config('weather')
config.d_model = 512
config.batch_size = 32
```

## 📖 更多文档

- **完整说明**: [README.md](README.md)
- **快速参考**: [QUICKSTART.md](QUICKSTART.md)
- **示例代码**: [examples/](examples/)
- **配置文件**: [config.py](config.py)
