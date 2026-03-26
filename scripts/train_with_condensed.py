"""
使用浓缩数据集训练模型

加载通过 get_condensed_dataset.py 生成的浓缩数据集，
用于训练 Autoformer、Transformer、Informer 等模型。

用法:
    # 使用默认配置训练
    python train_with_condensed.py --condensed_dir condensed_datasets/weather_kcenter_k500

    # 自定义配置
    python train_with_condensed.py \
        --condensed_dir condensed_datasets/weather_kcenter_k500 \
        --model Autoformer \
        --epochs 10 \
        --batch_size 32 \
        --learning_rate 0.0001
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time

from src.utils.feature_selection import CondensedDataset
from src.models import Autoformer, Transformer, Informer, Reformer
from src.utils.metrics import MSE, MAE
from src.utils import EarlyStopping
from src.exp.exp_main import Exp_Main


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用浓缩数据集训练模型')

    # 数据集参数
    parser.add_argument('--condensed_dir', type=str, required=True,
                        help='浓缩数据集目录路径')
    parser.add_argument('--data_path', type=str, default='dataset/weather.csv',
                        help='原始数据集路径（用于加载测试集）')
    parser.add_argument('--root_path', type=str, default='./',
                        help='根路径')

    # 序列参数（用于模型）
    parser.add_argument('--seq_len', type=int, default=336,
                        help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=96,
                        help='标签序列长度')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='预测长度')
    parser.add_argument('--freq', type=str, default='10min',
                        help='时间频率')

    # 模型参数
    parser.add_argument('--model', type=str, default='Autoformer',
                        choices=['Autoformer', 'Transformer', 'Informer', 'Reformer'],
                        help='模型类型')
    parser.add_argument('--enc_in', type=int, default=None,
                        help='编码器输入维度（从浓缩数据集自动获取）')
    parser.add_argument('--dec_in', type=int, default=None,
                        help='解码器输入维度')
    parser.add_argument('--c_out', type=int, default=None,
                        help='输出维度')
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='前馈网络维度')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='移动平均窗口')
    parser.add_argument('--factor', type=int, default=1,
                        help='注意力因子')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='激活函数')
    parser.add_argument('--output_attention', action='store_true',
                        help='是否输出注意力权重')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping 耐心值')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='是否使用学习率衰减')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='学习率衰减因子')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='学习率衰减耐心值')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='是否使用 GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU 设备 ID')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='是否使用多 GPU')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='使用的 GPU 设备')

    return parser.parse_args()


def train_model(model, train_loader, val_loader, args, device):
    """训练模型"""
    print("\n开始训练...")
    print("=" * 70)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 学习率调度器
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-7
        )
        print(f"学习率衰减已启用: factor={args.lr_factor}, patience={args.lr_patience}")

    # 从元数据获取参数
    metadata = train_loader.dataset.metadata
    seq_len = metadata.get('seq_len', 96)
    label_len = metadata.get('label_len', 96)
    pred_len = metadata.get('pred_len', 96)
    features = metadata.get('features', 'M')

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # ========== 训练 ==========
        model.train()
        train_loss = []

        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # 构造解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).to(device)

            # 前向传播
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # 计算损失
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)

        # ========== 验证 ==========
        model.eval()
        val_loss = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).to(device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_time

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")

        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping
        checkpoint_dir = './checkpoints/condensed_checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        early_stopping(val_loss, model, path=checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses


def test_model(model, test_loader, args, device):
    """测试模型"""
    print("\n测试模型...")
    print("=" * 70)

    model.eval()
    preds = []
    trues = []

    # 从参数获取序列配置
    seq_len = getattr(args, 'seq_len', 96)
    label_len = getattr(args, 'label_len', 96)
    pred_len = getattr(args, 'pred_len', 96)
    features = getattr(args, 'features', 'M')

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae = MAE(preds, trues)
    mse = MSE(preds, trues)

    return mae, mse


def main():
    """主函数"""
    args = parse_args()

    # 设备配置
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"模型: {args.model}")
    print(f"浓缩数据集: {args.condensed_dir}")
    print()

    # 加载浓缩数据集
    condensed_dataset = CondensedDataset(args.condensed_dir)
    metadata = condensed_dataset.metadata

    print(f"数据集大小: {len(condensed_dataset)}")
    print(f"算法: {metadata.get('algorithm', 'N/A')}")
    print(f"特征维度: {metadata.get('d_model', 'N/A')}")
    print(f"覆盖半径: {metadata.get('radius', 'N/A'):.4f}")
    print(f"原始大小: {metadata.get('original_size', 'N/A')}")
    print(f"压缩比例: {metadata.get('compression_ratio', 'N/A'):.2f}%")
    print(f"序列配置: seq_len={metadata.get('seq_len', 'N/A')}, "
          f"pred_len={metadata.get('pred_len', 'N/A')}")
    print()

    # 自动获取维度
    if args.enc_in is None:
        args.enc_in = condensed_dataset.data_x.shape[-1]
        print(f"自动设置 enc_in = {args.enc_in}")
    if args.dec_in is None:
        args.dec_in = args.enc_in
    if args.c_out is None:
        args.c_out = args.enc_in

    # 从元数据获取序列参数
    args.seq_len = metadata.get('seq_len', args.seq_len)
    args.label_len = metadata.get('label_len', args.label_len)
    args.pred_len = metadata.get('pred_len', args.pred_len)
    args.freq = metadata.get('freq', args.freq)
    print(f"序列参数: seq_len={args.seq_len}, label_len={args.label_len}, pred_len={args.pred_len}, freq={args.freq}")

    # 创建 DataLoader
    train_loader = DataLoader(
        condensed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    print(f"训练集批量数: {len(train_loader)}")
    print()

    # 加载验证集和测试集（使用原始数据集）
    from src.data.data_loader import Dataset_Custom

    val_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        size=[metadata.get('seq_len', 96),
              metadata.get('label_len', 96),
              metadata.get('pred_len', 96)],
        features=metadata.get('features', 'M'),
        target=metadata.get('target', 'OT'),
        scale=True,
        timeenc=0,
        freq=metadata.get('freq', '10min')
    )

    test_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='test',
        size=[metadata.get('seq_len', 96),
              metadata.get('label_len', 96),
              metadata.get('pred_len', 96)],
        features=metadata.get('features', 'M'),
        target=metadata.get('target', 'OT'),
        scale=True,
        timeenc=0,
        freq=metadata.get('freq', '10min')
    )

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"验证集大小: {len(val_set)}")
    print(f"测试集大小: {len(test_set)}")
    print()

    # 创建模型
    # 选择模型
    if args.model == 'Autoformer':
        model = Autoformer
    elif args.model == 'Transformer':
        model = Transformer
    elif args.model == 'Informer':
        model = Informer
    elif args.model == 'Reformer':
        model = Reformer
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model(args).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model}")
    print(f"参数量: {params:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print()

    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, args, device
    )

    # 测试模型
    print("\n验证集评估:")
    val_mae, val_mse = test_model(model, val_loader, args, device)
    print(f"  Val MAE: {val_mae:.4f}, Val MSE: {val_mse:.4f}")

    print("\n测试集评估:")
    test_mae, test_mse = test_model(model, test_loader, args, device)
    print(f"  Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")

    # 添加标准格式的输出，便于pipeline提取
    print(f"\nFinal Results:")
    print(f"MSE: {test_mse}")
    print(f"MAE: {test_mae}")



if __name__ == '__main__':
    main()
