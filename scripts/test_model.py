"""
Test script for running Autoformer-family models with weather dataset.
Usage:
    python scripts/test_model.py --model Autoformer --seq_len 96 --pred_len 96
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import random
import numpy as np

from src.exp.exp_main import Exp_Main


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer-family models for Time Series Forecasting')

    # model
    parser.add_argument('--model', type=str, default='Autoformer',
                        choices=['Autoformer', 'Transformer', 'Informer', 'Reformer'],
                        help='model name')
    parser.add_argument('--is_training', type=int, default=1, help='1=train, 0=test')
    parser.add_argument('--itr', type=int, default=1, help='experiment iterations')

    # data
    parser.add_argument('--root_path', type=str, default='./', help='root path')
    parser.add_argument('--data_path', type=str, default='dataset/weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'],
                        help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='10min', help='time feature frequency')

    # sequence
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=336, help='decoder start token length (should equal seq_len for Autoformer)')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

    # model architecture
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input dimension')
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input dimension')
    parser.add_argument('--c_out', type=int, default=21, help='output dimension')
    parser.add_argument('--d_model', type=int, default=512, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--moving_avg', type=int, default=25, help='moving average window')
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--distil', type=bool, default=True, help='use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='time embedding type')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_attention', type=bool, default=False, help='output attention weights')

    # reformer-specific
    parser.add_argument('--bucket_size', type=int, default=4, help='Reformer bucket size')
    parser.add_argument('--n_hashes', type=int, default=4, help='Reformer n_hashes')

    # training
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--des', type=str, default='test', help='experiment description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment')
    parser.add_argument('--use_amp', type=bool, default=False, help='automatic mixed precision')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0', help='GPU device ids')

    # time encoding
    parser.add_argument('--timeenc', type=int, default=0, help='0=manual dt, 1=time_features')

    # misc
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    print(f'Model: {args.model}')
    print(f'Data: {args.data_path}')
    print(f'seq_len={args.seq_len}, label_len={args.label_len}, pred_len={args.pred_len}')
    print(f'features={args.features}, target={args.target}, freq={args.freq}')
    print(f'd_model={args.d_model}, n_heads={args.n_heads}, e_layers={args.e_layers}, d_layers={args.d_layers}')
    print(f'use_gpu={args.use_gpu}')

    Exp = Exp_Main

    setting = f'weather_{args.model}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_{args.des}'

    exp = Exp(args)

    if args.is_training:
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
    else:
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

    torch.cuda.empty_cache()
    print('Done!')


if __name__ == '__main__':
    main()
