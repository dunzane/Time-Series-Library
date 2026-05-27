#!/bin/bash
# Resume-safe: finished runs are skipped by .done flags.
# High-Momentum & Stabilized Configuration for Extreme Multi-channel Traffic Flow (Diffmax × Traffic)
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

BACKBONE="PatchTST"
DATASET="Traffic"

LOGS_DIR="./logs/diffmax_pl192_patchtst_traffic"
DONE_DIR="./done/diffmax_pl192_patchtst_traffic"

mkdir -p "${LOGS_DIR}" "${DONE_DIR}"

SEQ_LEN=336
LABEL_LEN=48

PRED_LENS=(192)

SEEDS=(2021 2022 2023)

LRS=(0.0002)

ALPHAS=(0.60 0.7 0.8 0.9)

run_one() {
  local normalizer=$1
  local pred_len=$2
  local seed=$3
  local lr=$4
  local alpha=${5:-none}

  local run_id
  local model_id
  local des
  local norm_args

  if [[ "${normalizer}" == "softmax" ]]; then
    run_id="${BACKBONE}_${DATASET}_sl${SEQ_LEN}_pl${pred_len}_softmax_seed${seed}_lr${lr}"
    model_id="traffic_${SEQ_LEN}_${pred_len}_softmax_lr${lr}_seed${seed}"
    des="softmax_lr${lr}_seed${seed}"
    norm_args="--normalizer softmax"
  else
    run_id="${BACKBONE}_${DATASET}_sl${SEQ_LEN}_pl${pred_len}_diffmax_a${alpha}_seed${seed}_lr${lr}"
    model_id="traffic_${SEQ_LEN}_${pred_len}_diffmax_a${alpha}_lr${lr}_seed${seed}"
    des="diffmax_a${alpha}_lr${lr}_seed${seed}"
    norm_args="--normalizer diffmax --diffmax_alpha ${alpha} --diffmax_n_iter 50"
  fi

  local done_flag="${DONE_DIR}/${run_id}.done"
  local log_file="${LOGS_DIR}/${run_id}.log"

  if [[ -f "${done_flag}" ]]; then
    echo "[SKIP] ${run_id}"
    return 0
  fi

  echo "[RUN ] ${run_id}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id "${model_id}" \
    --model "${BACKBONE}" \
    --data custom \
    --features M \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 512 \
    --d_ff 512 \
    --des "${des}" \
    --batch_size 4 \
    --learning_rate ${lr} \
    --train_epochs 2 \
    --patience 3 \
    --loss mae \
    --seed ${seed} \
    ${norm_args} \
    --itr 1 \
    > "${log_file}" 2>&1

  if [[ $? -eq 0 ]]; then
    touch "${done_flag}"
    echo "[DONE] ${run_id}"
  else
    echo "[FAIL] ${run_id}"
    echo "       log: ${log_file}"
  fi
}

for pred_len in "${PRED_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do

      run_one "softmax" "${pred_len}" "${seed}" "${lr}"

      for alpha in "${ALPHAS[@]}"; do
        run_one "diffmax" "${pred_len}" "${seed}" "${lr}" "${alpha}"
      done

    done
  done
done

wait

echo ""
echo "[${BACKBONE} × ${DATASET}] All Traffic smooth_l1 & high LR prioritized runs finished or skipped."
