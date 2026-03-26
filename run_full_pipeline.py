"""
Complete Training Pipeline

Steps:
1. Pretrain Transformer Encoder
2. Get condensed datasets using K-Center and Herding algorithms
3. Train models (e.g., Autoformer) with condensed datasets
4. Save results (MSE, MAE) as a table

Usage:
    python run_full_pipeline.py --data_path dataset/weather.csv --k 500

Note:
    If GPU-related errors occur during model training, stop execution
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import numpy as np
import pandas as pd
import subprocess
import json
import glob
import time
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Complete Training Pipeline')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='dataset/weather.csv',
                        help='Dataset path')
    parser.add_argument('--root_path', type=str, default='./',
                        help='Root path')
    parser.add_argument('--data_name', type=str, default='weather',
                        help='Dataset name (for saving results)')

    # Sequence parameters
    parser.add_argument('--seq_len', type=int, default=336,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=336,
                        help='Label sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')

    # PatchTST Encoder parameters
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length (small for fine-grained analysis)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride for patching (must be < patch_len for overlap)')
    parser.add_argument('--aggregation', type=str, default='max',
                        choices=['max', 'flatten'],
                        help='Aggregation strategy: max or flatten')
    parser.add_argument('--target_dim', type=int, default=256,
                        help='Target output dimension (for flatten aggregation)')
    parser.add_argument('--concat_rev_params', type=bool, default=True,
                        help='Whether to concatenate RevIN parameters')

    # Encoder parameters
    parser.add_argument('--c_in', type=int, default=21,
                        help='Input feature dimension')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer feature dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward network dimension')

    # Pretraining parameters
    parser.add_argument('--pretrain_epochs', type=int, default=30,
                        help='Number of pretraining epochs')
    parser.add_argument('--pretrain_batch_size', type=int, default=32,
                        help='Pretraining batch size')
    parser.add_argument('--pretrain_lr', type=float, default=0.001,
                        help='Pretraining learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio')

    # Condensed dataset parameters
    parser.add_argument('--k', type=int, default=500,
                        help='Number of samples to select')
    parser.add_argument('--algorithms', type=str, default='kcenter,herding',
                        help='Algorithms to use (comma-separated)')

    # Training parameters
    parser.add_argument('--models', type=str, default='Autoformer,Transformer,Informer',
                        help='Models to train (comma-separated)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='Whether to use learning rate decay')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Learning rate decay factor')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Learning rate decay patience')

    # Device parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pretraining (use existing encoder)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Pretrained encoder path (if skipping pretraining)')

    # Save parameters
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Results save directory')

    return parser.parse_args()


def run_command(cmd, description, error_stop=True):
    """
    Run shell command and handle errors

    Parameters
    ----------
    cmd : list
        Command list
    description : str
        Command description
    error_stop : bool
        Whether to stop execution on error

    Returns
    -------
    success : bool
        Whether successful
    output : str
        Output information
    """
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=360000  # 10 hour timeout
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"\n[OK] {description} completed")
        return True, result.stdout

    except subprocess.TimeoutExpired as e:
        print(f"\n[ERROR] Error: Command execution timeout")
        if error_stop:
            raise
        return False, str(e)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error: Command execution failed (return code: {e.returncode})")
        print(e.stdout)
        print(e.stderr)

        # Check if it's a GPU-related error
        error_msg = e.stderr.lower() if e.stderr else ""
        if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'out of memory', 'device']):
            # Make sure it's actually a GPU error, not just mentioning "cuda" in the path
            if 'runtime error' in error_msg and ('cuda' in error_msg or 'gpu' in error_msg):
                print("\n[WARNING] GPU-related error detected, stopping execution")
                if error_stop:
                    raise Exception("GPU error, stopping execution")

        return False, str(e)

    except Exception as e:
        print(f"\n[ERROR] Unknown error: {str(e)}")
        if error_stop:
            raise
        return False, str(e)


def pretrain_encoder(args):
    """Step 1: Pretrain PatchTST Encoder"""
    save_path = f'outputs/checkpoints/patchtst_pretrained_{args.data_name}.pth'

    cmd = [
        'python', 'scripts/pretrain_encoder.py',
        '--data_path', args.data_path,
        '--root_path', args.root_path,
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(args.pred_len),
        '--c_in', str(args.c_in),
        '--patch_len', str(args.patch_len),
        '--stride', str(args.stride),
        '--d_model', str(args.d_model),
        '--n_heads', str(args.n_heads),
        '--e_layers', str(args.e_layers),
        '--d_ff', str(args.d_ff),
        '--aggregation', args.aggregation,
        '--mask_ratio', str(args.mask_ratio),
        '--epochs', str(args.pretrain_epochs),
        '--batch_size', str(args.pretrain_batch_size),
        '--learning_rate', str(args.pretrain_lr),
        '--device', 'cuda',
        '--save_path', save_path
    ]

    success, output = run_command(cmd, "Step 1: Pretrain Transformer Encoder")

    if not success:
        raise Exception("Pretraining failed")

    # Return encoder_only path (with latest ID)
    base_encoder_path = save_path.replace('.pth', '_encoder_only.pth')
    encoder_path = get_latest_encoder_path(base_encoder_path)
    return encoder_path


def get_latest_encoder_path(base_path):
    """
    查找最新的带ID后缀的编码器文件

    Parameters
    ----------
    base_path : str
        基础路径（不含ID），如：outputs/checkpoints/patchtst_pretrained_xxx_encoder_only.pth

    Returns
    -------
    str
        最新编码器文件的完整路径，如：outputs/checkpoints/patchtst_pretrained_xxx_encoder_only_001.pth
    """
    save_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).replace('.pth', '')

    # 搜索匹配模式的文件
    pattern = f'{base_name}_encoder_only_*.pth'
    encoder_files = glob.glob(os.path.join(save_dir, pattern))

    if not encoder_files:
        raise FileNotFoundError(
            f"找不到匹配的编码器文件: {pattern}\n"
            f"搜索目录: {save_dir}\n"
            f"请确保预训练步骤已成功完成。"
        )

    # 提取ID并找到最新的
    def extract_id(filepath):
        try:
            # 从文件名中提取ID，如: base_encoder_only_001.pth -> 1
            id_part = filepath.replace(f'{base_name}_encoder_only_', '').replace('.pth', '')
            return int(id_part)
        except ValueError:
            return 0

    latest_file = max(encoder_files, key=extract_id)
    print(f"找到最新的编码器文件: {os.path.basename(latest_file)}")
    return latest_file


def get_condensed_datasets(args, encoder_path):
    """Step 2: Get condensed datasets using K-Center and Herding"""
    algorithms = args.algorithms.split(',')
    condensed_dirs = {}

    for idx, algorithm in enumerate(algorithms):
        algorithm = algorithm.strip()
        print(f"\n{'='*60}")
        print(f"Step 2.{idx+1}: Getting condensed dataset using {algorithm.upper()} algorithm")
        print(f"{'='*60}")

        save_dir = f'condensed_datasets/{args.data_name}_{algorithm}_k{args.k}'

        cmd = [
            'python', 'scripts/get_condensed_dataset.py',
            '--algorithm', algorithm,
            '--k', str(args.k),
            '--data_path', args.data_path,
            '--root_path', args.root_path,
            '--seq_len', str(args.seq_len),
            '--label_len', str(args.label_len),
            '--pred_len', str(args.pred_len),
            '--c_in', str(args.c_in),
            '--patch_len', str(args.patch_len),
            '--stride', str(args.stride),
            '--d_model', str(args.d_model),
            '--n_heads', str(args.n_heads),
            '--e_layers', str(args.e_layers),
            '--d_ff', str(args.d_ff),
            '--aggregation', args.aggregation,
            '--target_dim', str(args.target_dim),
            '--concat_rev_params', 'True' if args.concat_rev_params else 'False',
            '--pretrained_path', encoder_path,
            '--save_dir', save_dir,
            '--device', 'cuda'
        ]

        success, output = run_command(
            cmd,
            f"Step 2.{idx+1}: {algorithm.upper()} algorithm",
            error_stop=False
        )

        if success:
            condensed_dirs[algorithm] = save_dir
            print(f"\n✓ 成功: {algorithm.upper()}算法完成")
            print(f"  压缩数据集保存至: {save_dir}")
        else:
            print(f"\n✗ 失败: {algorithm.upper()}算法执行失败")
            print(f"  错误输出:\n{output}")
            print(f"  跳过{algorithm.upper()}算法，继续执行剩余算法...\n")

    # 检查是否有成功的数据集
    if not condensed_dirs:
        raise Exception("所有算法都失败了！请检查编码器路径和参数配置。")

    print(f"\n{'='*60}")
    print(f"成功创建 {len(condensed_dirs)} 个压缩数据集:")
    for alg, path in condensed_dirs.items():
        print(f"  - {alg.upper()}: {path}")
    print(f"{'='*60}\n")

    return condensed_dirs


def train_with_condensed(args, condensed_dirs, encoder_path):
    """Step 3: Train models with condensed datasets"""
    models = args.models.split(',')
    results = []

    for algorithm, save_dir in condensed_dirs.items():
        for model_name in models:
            model_name = model_name.strip()

            print(f"Training {model_name} with {algorithm.upper()} condensed dataset")

            # Modify training script to return results
            result = train_single_model(args, model_name, save_dir)

            if result is not None:
                results.append({
                    'Algorithm': algorithm.upper(),
                    'Model': model_name,
                    'K': args.k,
                    'MSE': result['mse'],
                    'MAE': result['mae'],
                    'RMSE': result['rmse'],
                    'Time': result['time']
                })
            else:
                print(f"Warning: {model_name} training failed")

    return results


def train_single_model(args, model_name, condensed_dir):
    """Train a single model and return results"""
    import time

    checkpoint_dir = f'./checkpoints/{args.data_name}_{model_name}_{Path(condensed_dir).name}'

    cmd = [
        'python', 'scripts/train_with_condensed.py',
        '--condensed_dir', condensed_dir,
        '--data_path', args.data_path,
        '--root_path', args.root_path,
        '--model', model_name,
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(args.pred_len),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--patience', str(args.patience),
        '--gpu', str(args.gpu),
        '--use_gpu', 'True'
    ]

    # If learning rate scheduler is enabled, add related parameters
    if args.lr_scheduler:
        cmd.append('--lr_scheduler')
        cmd.extend(['--lr_factor', str(args.lr_factor)])
        cmd.extend(['--lr_patience', str(args.lr_patience)])

    start_time = time.time()
    success, output = run_command(cmd, f"Train {model_name}", error_stop=False)
    elapsed_time = time.time() - start_time

    if not success:
        return None

    # Extract MSE and MAE from output
    mse = None
    mae = None

    for line in output.split('\n'):
        # 尝试多种格式
        if 'MSE:' in line or 'mse:' in line:
            try:
                # 处理 "MSE: 0.1234" 或 "Test MSE: 0.1234" 格式
                mse_str = line.split(':')[-1].strip()
                mse = float(mse_str)
            except:
                # 尝试从逗号分隔的字符串中提取
                try:
                    # 处理 "Test MAE: 0.1234, Test MSE: 0.5678" 格式
                    parts = line.split(',')
                    for part in parts:
                        if 'MSE:' in part or 'mse:' in part:
                            mse = float(part.split(':')[-1].strip())
                except:
                    pass
        if 'MAE:' in line or 'mae:' in line:
            try:
                mae_str = line.split(':')[-1].strip()
                mae = float(mae_str)
            except:
                try:
                    parts = line.split(',')
                    for part in parts:
                        if 'MAE:' in part or 'mae:' in part:
                            mae = float(part.split(':')[-1].strip())
                except:
                    pass

    if mse is None or mae is None:
        print("Warning: Unable to extract MSE/MAE from output")
        print("Output preview:")
        print(output[-500:] if len(output) > 500 else output)
        return None

    print(f"✓ 成功提取结果: MSE={mse:.4f}, MAE={mae:.4f}")

    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'time': elapsed_time
    }


def main():
    """Main function"""
    args = parse_args()


    print(f"Dataset: {args.data_path}")
    print(f"Number of samples K: {args.k}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Models: {args.models}")
    print(f"Pretraining epochs: {args.pretrain_epochs}")
    print(f"Training epochs: {args.epochs}")
    print(f"GPU: {args.gpu}")
    print()

    try:
        # Step 1: Pretrain Encoder
        if not args.skip_pretrain:
            encoder_path = pretrain_encoder(args)
        else:
            encoder_path = args.pretrained_path
            if encoder_path is None:
                raise ValueError("--pretrained_path must be provided when skipping pretraining")
            print(f"Using existing pretrained encoder: {encoder_path}")

        # Step 2: Get condensed datasets
        condensed_dirs = get_condensed_datasets(args, encoder_path)

        if not condensed_dirs:
            raise Exception("No condensed datasets were successfully generated")

        # Step 3: Train models
        results = train_with_condensed(args, condensed_dirs, encoder_path)

        # Step 4: Save results
        if results:
            pass
            # save_results(results, args)
        else:
            print("Warning: All model training failed")

    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
