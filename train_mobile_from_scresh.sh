#!/usr/bin/env sh

DATASET_DIR=./tf_records/
TRAIN_DIR=./logs_mobile/
CHECKPOINT_PATH=./logs_mobile/
python train_mobilenet_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=lisa \
    --dataset_split_name=train \
    --model_name=mobilenet_ssd_traffic \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --optimizer=adam \
    --learning_rate=0.005 \
    --batch_size=6
