#!/usr/bin/env sh

DATASET_DIR=./tf_records/
TRAIN_DIR=./logs_obj_lyj/
CHECKPOINT_PATH=./logs_obj_lyj/
python train_mobilenet_ssd_network_finetune_owndata_obj.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=smartphone_traffic \
    --dataset_split_name=train \
    --model_name=mobilenet_pretrained_owndata_obj \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --batch_size=12
