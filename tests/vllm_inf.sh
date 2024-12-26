#!/bin/bash

cd /data/chatgpt/data/baliao/psd/01_serve/psd

hostname --ip-address

export CUDA_VISIBLE_DEVICES=0
python test/generation.py