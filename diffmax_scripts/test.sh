#!/bin/bash
# Smoke test for diffmax-cross-backbone.
# Resume-safe: finished runs are skipped by .done flags.
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3

BACKBONE="PatchTST"
DATASET="ETTh1"

LOGS_DIR="./logs/test_diffmax_cross_backbone_patchtst_etth1"
DONE_DIR="./done/test_diffmax_cross_backbone_patchtst_etth1"

mkdir -p "${LOGS_DIR}" "${DONE_DIR}"

SEQ_LEN=336
LABEL_LEN=48

# Use the most memory-consuming horizon first.
PRED_LENS=(720)

# Minimal smoke-test grid.
SEEDS=(2021)
LRS=(0.0001)
ALPHAS=(0.30)

MAX_JOBS=1

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
    run_id="TEST_${BACKBONE}_${DATASET}_sl${SEQ_LEN}_pl${pred_len}_softmax_seed${seed}_lr${lr}"
    model_id="TEST_${DATASET}_${SEQ_LEN}_${pred_len}_softmax_lr${lr}_seed${seed}"
    des="test_softmax_lr${lr}_seed${seed}"
    norm_args="--normalizer softmax"
  else
    run_id="TEST_${BACKBONE}_${DATASET}_sl${SEQ_LEN}_pl${pred_len}_diffmax_a${alpha}_seed${seed}_lr${lr}"
    model_id="TEST_${DATASET}_${SEQ_LEN}_${pred_len}_diffmax_a${alpha}_lr${lr}_seed${seed}"
    des="test_diffmax_a${alpha}_lr${lr}_seed${seed}"
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
    --data_path ETTh1.csv \
    --model_id "${model_id}" \
    --model "${BACKBONE}" \
    --data ETTh1 \
    --features M \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${pred_len} \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des "${des}" \
    --n_heads 2 \
    --learning_rate ${lr} \
    --seed ${seed} \
    --train_epochs 1 \
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
echo "[TEST ${BACKBONE} × ${DATASET}] All scheduled runs finished or skipped."