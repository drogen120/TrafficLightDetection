#!/usr/bin/env sh

DATASET_DIR=./tf_records/
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./logs/
python train_mobilenet_ssd_network_finetune.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=lisa \
    --dataset_split_name=train \
    --model_name=mobilenet_pretrained \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --batch_size=6
