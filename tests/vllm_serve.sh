#!/bin/bash

cd /data/chatgpt/data/baliao/psd/01_serve/psd

hostname --ip-address

export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
        --model /mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B-Instruct \
        --served-model-name Qwen2.5-Math-1.5B-Instruct \
        --tensor-parallel-size 1 \
        --port 8888 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-num-batched-tokens 99999 \
        --max-num-seqs 9999 \
        --max-model-len 4096 \
        --enforce-eager