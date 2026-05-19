#!/bin/bash

models=("gcn" "ngcf" "lrgccf" "mf" "neumf" "caged" "gtn" "simgcl")
SEED=42
LOG_DIR="./log/kuairec"
mkdir -p $LOG_DIR

for model in "${models[@]}"
do
    echo "Starting evaluation for model: $model"
    python judge_kuairec.py --model "$model" --seed "$SEED" > "$LOG_DIR/${model}_eval_full_${SEED}.log" 2>&1
    echo "Finished $model. Logs saved to $LOG_DIR/${model}_eval_full_${SEED}.log"
done

echo "All models finished."