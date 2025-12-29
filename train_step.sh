#!/bin/bash

# Load environment variables
export HF_HOME="/data2/models/"
export TRANSFORMERS_CACHE="/data2/models/"
export HF_DATASETS_CACHE="/data2/datasets/"
export TOKENIZERS_PARALLELISM=false

# Step-based validation and saving settings
VAL_EVERY_N_STEPS=30
SAVE_EVERY_N_STEPS=30

CUDA_VISIBLE_DEVICES=4 python train_llama.py \
    --model_name meta-llama/Llama-3.2-1B \
    --save_path ./checkpoints/llama_3.2_1b_meetingbank_step \
    --batch_size 4 \
    --num_epoch 100 \
    --gradient_accumulation_steps 64 \
    --val_every_n_steps $VAL_EVERY_N_STEPS \
    --save_every_n_steps $SAVE_EVERY_N_STEPS
