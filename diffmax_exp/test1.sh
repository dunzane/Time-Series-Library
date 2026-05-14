#!/bin/bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

model_name=PatchTST

SEEDS=(2021 2022 2023)
LRS=(0.00005 0.00002 0.0001)
ALPHAS=(0.1 0.2 0.3)

GPUS=(0)
job_id=0

run_softmax () {
  local seed=$1
  local lr=$2
  local gpu=$3

  CUDA_VISIBLE_DEVICES=${gpu} python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_96_raw_softmax_lr${lr}_seed${seed} \
    --model ${model_name} \
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
    --des raw_softmax_lr${lr}_seed${seed} \
    --n_heads 2 \
    --learning_rate ${lr} \
    --seed ${seed} \
    --normalizer softmax \
    --itr 1
}

run_diffmax () {
  local seed=$1
  local lr=$2
  local alpha=$3
  local gpu=$4

  CUDA_VISIBLE_DEVICES=${gpu} python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_96_raw_diffmax_a${alpha}_lr${lr}_seed${seed} \
    --model ${model_name} \
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
    --des raw_diffmax_a${alpha}_lr${lr}_seed${seed} \
    --n_heads 2 \
    --learning_rate ${lr} \
    --seed ${seed} \
    --normalizer diffmax \
    --diffmax_alpha ${alpha} \
    --itr 1
}

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do

    gpu=${GPUS[$((job_id % ${#GPUS[@]}))]}
    run_softmax ${seed} ${lr} ${gpu} &
    job_id=$((job_id + 1))

    if (( job_id % ${#GPUS[@]} == 0 )); then
      wait
    fi

    for alpha in "${ALPHAS[@]}"; do
      gpu=${GPUS[$((job_id % ${#GPUS[@]}))]}
      run_diffmax ${seed} ${lr} ${alpha} ${gpu} &
      job_id=$((job_id + 1))

      if (( job_id % ${#GPUS[@]} == 0 )); then
        wait
      fi
    done

  done
done

wait