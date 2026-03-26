# 快速参考指南

## 常用命令

### 基础测试

```bash
# 测试所有模型
python examples/01_basic_test.py

# 基于特征的数据选择示例
python examples/feature_selection_example.py
```

### 训练模型

```bash
# 使用默认配置训练
python run.py --model Autoformer --is_training 1

# 使用自定义参数
python run.py --model Autoformer --is_training 1 \
    --seq_len 96 --pred_len 96 \
    --d_model 512 --batch_size 32
```

### 测试模型

```bash
# 加载 checkpoint 测试
python run.py --model Autoformer --is_training 0
```

## 快速配置

### 方法 1：使用配置文件

```python
# config.py
from config import get_config

config = get_config('weather')  # base, small, large, weather

# 使用配置
python run.py @config.py --model Autoformer --is_training 1
```

### 方法 2：命令行参数

```bash
python run.py \
    --model Autoformer \
    --data_path dataset/weather.csv \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 512 \
    --batch_size 32 \
    --train_epochs 10
```

## 参数速查表

### 数据相关

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `--data_path` | 数据集路径 | `dataset/weather.csv` |
| `--features` | 预测模式 | `M` (多变量), `S` (单变量) |
| `--target` | 目标列 | `OT` |
| `--freq` | 时间频率 | `10min`, `h`, `d` |
| `--seq_len` | 输入长度 | 96, 168, 720 |
| `--pred_len` | 预测长度 | 24, 96, 192 |

### 模型相关

| 参数 | 说明 | 快速 | 标准 | 大型 |
|------|------|------|------|------|
| `--d_model` | 模型维度 | 128 | 512 | 1024 |
| `--n_heads` | 注意力头 | 4 | 8 | 16 |
| `--e_layers` | 编码器层数 | 2 | 2 | 4 |
| `--d_ff` | FFN 维度 | 256 | 2048 | 4096 |

### 训练相关

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--batch_size` | 批量大小 | 32 (GPU 内存不足时减小) |
| `--learning_rate` | 学习率 | 0.0001 |
| `--train_epochs` | 训练轮数 | 10-50 |
| `--patience` | Early stopping | 3 |

## 常见使用场景

### 场景 1：快速验证

```bash
python run.py --model Autoformer --is_training 1 \
    --d_model 128 --e_layers 1 --batch_size 64 --train_epochs 2
```

### 场景 2：标准训练

```bash
python run.py --model Autoformer --is_training 1 \
    --d_model 512 --n_heads 8 --e_layers 2 \
    --batch_size 32 --train_epochs 10
```

### 场景 3：长序列预测

```bash
python run.py --model Autoformer --is_training 1 \
    --seq_len 720 --pred_len 192 \
    --d_model 512 --e_layers 3
```

### 场景 4：使用特征选择浓缩数据集训练

```python
# examples/feature_selection_example.py
from feature_selection import select_samples
from layers.FeatureEncoder import create_feature_encoder

# 创建特征编码器
encoder = create_feature_encoder(c_in=21, d_model=128)

# 在特征空间选择 500 个样本
selected_indices, radius = select_samples(
    train_loader,
    encoder,
    k=500,
    algorithm='kcenter'
)
```

## 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 长序列预测 | Autoformer | O(L log L) 复杂度，自相关机制 |
| 中短序列 | Transformer | O(L²) 复杂度，标准注意力 |
| 内存受限 | Informer | ProbSparse 注意力 |
| 超长序列 | Reformer | LSH 注意力 |

## 故障排查

### 内存不足

```bash
# 减小批量大小
python run.py --batch_size 16

# 减小模型维度
python run.py --d_model 256 --d_ff 512
```

### 训练慢

```bash
# 使用 GPU
python run.py --use_gpu True

# 减少数据量（使用 K-Center）
# 参考 examples/02_kcenter_selection.py
```

### 结果不理想

```bash
# 增加模型容量
python run.py --d_model 1024 --e_layers 4

# 调整学习率
python run.py --learning_rate 0.00001

# 增加 epochs
python run.py --train_epochs 50
```

## 输出文件

训练完成后会生成：

```
checkpoints/
└── weather_Autoformer_.../
    └── checkpoint.pth      # 模型权重

results/
└── weather_Autoformer_.../
    ├── metrics.npy         # [mae, mse, rmse, mape, mspe]
    ├── pred.npy           # 预测结果
    └── true.npy           # 真实值

result.txt                 # 所有实验的指标汇总
```

## 更多帮助

- 完整文档：查看 [README.md](README.md)
- 示例代码：查看 [examples/](examples/) 目录
