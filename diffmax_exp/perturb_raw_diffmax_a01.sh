#!/bin/bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MODEL_NAME=PatchTST
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}
SAVE_DIR=./perturb_results/raw_diffmax_a01
SEED=2021
LR=0.0001
ALPHA=0.1
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}

NOISES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ATTENS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p "${SAVE_DIR}"

for noise in "${NOISES[@]}"; do
  for atten in "${ATTENS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_336_96_raw_diffmax_a${ALPHA}_noise${noise}_atten${atten}_seed${SEED} \
      --model ${MODEL_NAME} \
      --data ETTh1 \
      --features M \
      --seq_len 336 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des raw_diffmax_a01_perturb \
      --n_heads 2 \
      --train_epochs ${TRAIN_EPOCHS} \
      --learning_rate ${LR} \
      --seed ${SEED} \
      --use_norm 0 \
      --normalizer diffmax \
      --diffmax_alpha ${ALPHA} \
      --attn_noise_scale ${noise} \
      --attn_attenuation ${atten} \
      --perturb_save_dir ${SAVE_DIR} \
      --perturb_tag raw_diffmax_a01 \
      --itr 1
  done
done
