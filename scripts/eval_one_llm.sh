#!/bin/bash
cd /data/chatgpt/data/baliao/psd/01_serve/psd

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
IP_ADDRESS="http://localhost:1234/v1"

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math500"
TOKENIZERS_PARALLELISM=false \
python3 -u one_llm.py \
    --data_name ${DATA_NAME} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --ip_address ${IP_ADDRESS} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \