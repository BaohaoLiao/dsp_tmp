## Installation
Already have a base docker image with transformers, vllm and so on.

```
FROM add_your_base_image
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip

# For Qwen2.5 math evaluation
RUN pip install --no-cache-dir \
       sympy \
       antlr4-python3-runtime==4.11.1 \
       word2number \
       Pebble \
       timeout-decorator

# For skywork PRM
RUN git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git && \
    cd skywork-o1-prm-inference && \
    pip install -e . && \
    rm -rf /skywork-o1-prm-inference/.git
CMD ["/bin/bash"]
```

## Key files or functions
- **get_responses** in psd.py

## Run
**Note**: vllm only supports one LLM at one GPU. One needs at least 3 GPUs to run the following code.
```
# Serve LLM1, LLM2 and PRM. Remember to use your local saved models or huggingface's.
bash scripts/vllm_serve_llm1.sh
bash scripts/vllm_serve_llm2.sh
bash scripts/vllm_serve_prm.sh

# Replace 'localhost' in scripts/eval_psd.sh with the ip addresses from the above serving
# Run eval, you should obtain an accuracy about 74 for math500
bash scripts/eval_psd.sh
```