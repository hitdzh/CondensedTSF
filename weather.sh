#!/bin/bash

# CondensedTSF Pipeline - Weather Dataset

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# ============= 配置参数 =============
GPU_ID=0
DATA_NAME=weather
ALGORITHMS=kcenter,herding
K=2000

# ============= 跳过预训练配置 =============
SKIP_PRETRAIN=False
# 当 SKIP_PRETRAIN=True 时，指定已训练的编码器路径
PRETRAINED_PATH=

# ============= 预训练参数 =============
SEQ_LEN=336
LABEL_LEN=96
PRED_LEN=96
C_IN=21
PATCH_LEN=16
STRIDE=8
D_MODEL=128
N_HEADS=8
E_LAYERS=3
D_FF=512
AGGREGATION=max
TARGET_DIM=256
CONCAT_REV_PARAMS=True

# ============= 训练参数 =============
EPOCHS=100
BATCH_SIZE=256
LEARNING_RATE=0.0001
PRETRAIN_EPOCHS=30
PRETRAIN_BATCH_SIZE=256
PRETRAIN_LR=0.001
MASK_RATIO=0.75
PATIENCE=10

# ============= 路径配置 =============
ROOT_PATH=./dataset/
DATA_PATH=weather.csv

# ============= 日志配置 =============
LOG_FILE=logs/${DATA_NAME}_${ALGORITHMS}_k${K}.log

# ============= 执行 =============
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 构建参数
PRETRAIN_ARGS=""
if [ "$SKIP_PRETRAIN" = "True" ] || [ "$SKIP_PRETRAIN" = "true" ]; then
    if [ -z "$PRETRAINED_PATH" ]; then
        echo "错误: SKIP_PRETRAIN=True 时必须指定 PRETRAINED_PATH"
        exit 1
    fi
    PRETRAIN_ARGS="--skip_pretrain --pretrained_path $PRETRAINED_PATH"
fi

python -u run_full_pipeline.py \
    --data_name $DATA_NAME \
    --data_path $DATA_PATH \
    --root_path $ROOT_PATH \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --c_in $C_IN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --e_layers $E_LAYERS \
    --d_ff $D_FF \
    --aggregation $AGGREGATION \
    --target_dim $TARGET_DIM \
    --concat_rev_params $CONCAT_REV_PARAMS \
    --k $K \
    --algorithms $ALGORITHMS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --pretrain_batch_size $PRETRAIN_BATCH_SIZE \
    --pretrain_lr $PRETRAIN_LR \
    --mask_ratio $MASK_RATIO \
    --patience $PATIENCE \
    --gpu $GPU_ID \
    $PRETRAIN_ARGS > $LOG_FILE 2>&1
