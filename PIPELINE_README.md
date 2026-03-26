# 完整训练流程 Pipeline 使用说明

## 概述

`run_full_pipeline.py` 实现了完整的时序预测模型训练流程：

1. **预训练 Encoder** - 使用 Masked Reconstruction 方法预训练 Transformer Encoder
2. **获取浓缩数据集** - 分别使用 K-Center 和 Herding 算法获取浓缩数据集
3. **训练模型** - 使用浓缩数据集训练 Autoformer 等模型
4. **保存结果** - 将结果 (MSE, MAE) 保存为表格

## 快速开始

### 基本用法

```bash
# 使用默认参数运行完整流程
python run_full_pipeline.py --data_path dataset/weather.csv

# 自定义样本数量
python run_full_pipeline.py --data_path dataset/weather.csv --k 500

# 指定使用的算法和模型
python run_full_pipeline.py \
    --data_path dataset/weather.csv \
    --k 500 \
    --algorithms kcenter,herding \
    --models Autoformer,Transformer
```

### 主要参数

#### 数据参数
- `--data_path`: 数据集路径 (默认: `dataset/weather.csv`)
- `--data_name`: 数据集名称 (默认: `weather`)
- `--root_path`: 根路径 (默认: `./`)
- `--seq_len`: 输入序列长度 (默认: 96)
- `--pred_len`: 预测长度 (默认: 96)
- `--c_in`: 输入特征维度 (默认: 21)

#### 预训练参数
- `--pretrain_epochs`: 预训练轮数 (默认: 30)
- `--pretrain_batch_size`: 预训练批量大小 (默认: 32)
- `--pretrain_lr`: 预训练学习率 (默认: 0.001)
- `--mask_ratio`: Mask 比例 (默认: 0.75)
- `--skip_pretrain`: 跳过预训练 (使用已有的编码器)

#### 浓缩数据集参数
- `--k`: 选择的样本数量 (默认: 500)
- `--algorithms`: 使用的算法，逗号分隔 (默认: `kcenter,herding`)

#### 训练参数
- `--models`: 训练的模型，逗号分隔 (默认: `Autoformer,Transformer,Informer`)
- `--epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批量大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 0.0001)

#### 设备参数
- `--gpu`: GPU 设备 ID (默认: 0)
- `--save_dir`: 结果保存目录 (默认: `results`)

## 示例

### 示例1: 快速测试（小规模）

```bash
python run_full_pipeline.py \
    --data_path dataset/weather.csv \
    --data_name weather_test \
    --k 100 \
    --pretrain_epochs 5 \
    --epochs 3 \
    --models Autoformer \
    --algorithms kcenter
```

### 示例2: 完整实验

```bash
python run_full_pipeline.py \
    --data_path dataset/weather.csv \
    --data_name weather_full \
    --k 500 \
    --pretrain_epochs 30 \
    --epochs 10 \
    --models Autoformer,Transformer,Informer,Reformer \
    --algorithms kcenter,herding \
    --gpu 0
```

### 示例3: 跳过预训练（使用已有的编码器）

```bash
python run_full_pipeline.py \
    --data_path dataset/weather.csv \
    --k 500 \
    --skip_pretrain \
    --pretrained_path outputs/checkpoints/encoder_pretrained_weather_encoder_only.pth \
    --epochs 10
```

## 输出结果

Pipeline 执行完成后，结果将保存在 `results/` 目录下：

- `results_weather_20240322_143020.csv` - CSV 格式结果表格
- `results_weather_20240322_143020.xlsx` - Excel 格式结果表格
- `results_weather_20240322_143020.json` - JSON 格式结果

### 结果表格格式

| Algorithm | Model | K | MSE | MAE | RMSE | Time |
|-----------|-------|---|-----|-----|------|------|
| KCENTER | Autoformer | 500 | 0.1234 | 0.2345 | 0.3512 | 120.5 |
| HERDING | Autoformer | 500 | 0.1245 | 0.2356 | 0.3528 | 118.3 |
| KCENTER | Transformer | 500 | 0.1256 | 0.2367 | 0.3544 | 145.2 |
| ... | ... | ... | ... | ... | ... | ... |

## 错误处理

- 如果训练过程中出现 GPU 相关错误（如显存不足），Pipeline 会自动停止执行
- 每个步骤的成功/失败都会在控制台输出
- 失败的算法或模型会被跳过，继续执行其他任务

## 单独使用各步骤

如果需要单独运行某个步骤：

### 1. 预训练 Encoder

```bash
python scripts/pretrain_encoder.py \
    --data_path dataset/weather.csv \
    --epochs 30 \
    --batch_size 32 \
    --mask_ratio 0.75
```

### 2. 获取浓缩数据集

```bash
# K-Center
python scripts/get_condensed_dataset.py \
    --algorithm kcenter \
    --k 500 \
    --pretrained_path outputs/checkpoints/encoder_pretrained_weather_encoder_only.pth

# Herding
python scripts/get_condensed_dataset.py \
    --algorithm herding \
    --k 500 \
    --pretrained_path outputs/checkpoints/encoder_pretrained_weather_encoder_only.pth
```

### 3. 训练模型

```bash
python scripts/train_with_condensed.py \
    --condensed_dir condensed_datasets/weather_kcenter_k500 \
    --model Autoformer \
    --epochs 10 \
    --batch_size 32
```

## 目录结构

执行完成后，项目目录结构：

```
CondensedTSF/
├── dataset/
│   └── weather.csv
├── condensed_datasets/
│   ├── weather_kcenter_k500/
│   │   ├── data_x.npy
│   │   ├── data_y.npy
│   │   ├── data_x_mark.npy
│   │   ├── data_y_mark.npy
│   │   ├── indices.npy
│   │   └── metadata.pkl
│   └── weather_herding_k500/
│       └── ...
├── outputs/
│   └── checkpoints/
│       ├── encoder_pretrained_weather.pth
│       └── encoder_pretrained_weather_encoder_only.pth
├── checkpoints/
│   └── weather_*_checkpoints/
├── results/
│   ├── results_weather_20240322_143020.csv
│   ├── results_weather_20240322_143020.xlsx
│   └── results_weather_20240322_143020.json
└── run_full_pipeline.py
```

## 注意事项

1. **GPU 内存**: 确保有足够的 GPU 内存。如果遇到 OOM (Out of Memory) 错误，可以：
   - 减小 `--batch_size`
   - 减小 `--d_model`
   - 减小 `--k` (样本数量)

2. **训练时间**: 完整流程可能需要较长时间，建议：
   - 先用小参数测试 (`--k 100`, `--epochs 3`)
   - 确认无误后再运行完整实验

3. **数据集**: 确保 CSV 数据集格式正确，第一列为日期时间

4. **依赖**: 确保安装了所有依赖包：
   ```bash
   pip install torch numpy pandas tqdm scikit-learn openpyxl
   ```
