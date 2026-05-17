#!/bin/bash
# Resume-safe: finished runs are skipped by .done flags.
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3

BACKBONE="Crossformer"
DATASET="ETTh2"

LOGS_DIR="./logs/diffmax_cross_backbone_crossformer_etth2"
DONE_DIR="./done/diffmax_cross_backbone_crossformer_etth2"

mkdir -p "${LOGS_DIR}" "${DONE_DIR}"

SEQ_LEN=336
LABEL_LEN=48

PRED_LENS=(720 336 192 96)
SEEDS=(2021 2022 2023)
LRS=(0.0001 0.00005 0.00002)
ALPHAS=(0.70 0.50 0.30 0.20 0.10)

MAX_JOBS=2

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 5
  done
}

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
    model_id="${DATASET}_${SEQ_LEN}_${pred_len}_cross_backbone_softmax_lr${lr}_seed${seed}"
    des="cross_backbone_softmax_lr${lr}_seed${seed}"
    norm_args="--normalizer softmax"
  else
    run_id="${BACKBONE}_${DATASET}_sl${SEQ_LEN}_pl${pred_len}_diffmax_a${alpha}_seed${seed}_lr${lr}"
    model_id="${DATASET}_${SEQ_LEN}_${pred_len}_cross_backbone_diffmax_a${alpha}_lr${lr}_seed${seed}"
    des="cross_backbone_diffmax_a${alpha}_lr${lr}_seed${seed}"
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
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id "${model_id}" \
    --model "${BACKBONE}" \
    --data ETTh2 \
    --features M \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des "${des}" \
    --learning_rate ${lr} \
    --seed ${seed} \
    --num_workers 0 \
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

      wait_for_slot
      run_one "softmax" "${pred_len}" "${seed}" "${lr}" &

      for alpha in "${ALPHAS[@]}"; do
        wait_for_slot
        run_one "diffmax" "${pred_len}" "${seed}" "${lr}" "${alpha}" &
      done

    done
  done
done

wait

echo ""
echo "[${BACKBONE} × ${DATASET}] All scheduled runs finished or skipped."