#!/bin/bash
cd /data/chatgpt/data/baliao/psd/01_serve/psd

PROMPT_TYPE="qwen25-math-cot"
LLM1_NAME_OR_PATH="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B-Instruct"
LLM2_NAME_OR_PATH="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B-Instruct"
PRM_NAME_OR_PATH="/mnt/nushare2/data/baliao/PLLMs/skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
OUTPUT_DIR="llm1_Qwen2.5-Math-1.5B-Instruct_llm2_Qwen2.5-Math-1.5B-Instruct_prm_Skywork-o1-Open-PRM-Qwen-2.5-1.5B/math_eval"
LLM1_IP_ADDRESS="http://localhost:12341/v1"
LLM2_IP_ADDRESS="http://localhost:12342/v1"
PRM_IP_ADDRESS="http://localhost:12343/v1"

SPLIT="test"
NUM_TEST_SAMPLE=-1
PRM_THRESHOLD=0.5

# English open datasets
DATA_NAME="math500"
TOKENIZERS_PARALLELISM=false \
python3 -u psd.py \
    --data_name ${DATA_NAME} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --llm1_name_or_path ${LLM1_NAME_OR_PATH} \
    --llm2_name_or_path ${LLM2_NAME_OR_PATH} \
    --prm_name_or_path ${PRM_NAME_OR_PATH} \
    --llm1_ip_address ${LLM1_IP_ADDRESS} \
    --llm2_ip_address ${LLM2_IP_ADDRESS} \
    --prm_ip_address ${PRM_IP_ADDRESS} \
    --max_prm_threshold ${PRM_THRESHOLD} \
    --min_prm_threshold ${PRM_THRESHOLD} \
    --max_turns 30 \
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