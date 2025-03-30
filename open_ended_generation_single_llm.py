import os
import argparse
import json
import time
from transformers import AutoTokenizer
from openai import OpenAI
from external.qwen25_math_evaluation.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--llm_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--llm_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--step_word", type=str, default="\n\n")
    parser.add_argument("--max_prm_threshold", type=float, default=0.5)
    parser.add_argument("--min_prm_threshold", type=float, default=None)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.min_prm_threshold = (
        args.max_prm_threshold if args.min_prm_threshold is None else args.min_prm_threshold
    )
    return args


PROMPT=(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def prepare_data(data_name, args):
    file_path = os.path.join(args.data_dir, data_name, args.split + ".json")
    with open(file_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)

    examples = []
    for sample in samples:
        sample["prompt"] = PROMPT.format(input=sample["instruction"])
        examples.append(sample)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_maxprm{args.max_prm_threshold}_minprm{args.min_prm_threshold}_maxturns{args.max_turns}.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file


def setup(args):
    # load model
    openai_api_key = "EMPTY"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=args.llm_ip_address,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name_or_path, trust_remote_code=True)

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        main(client, tokenizer, data_name, args)


def get_responses(args, client, tokenizer, prompts):
    responses = client.completions.create(
            model=args.llm_name_or_path.split("/")[-1],
            prompt=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
        ).choices
    responses = sorted(responses, key=lambda x: int(x.index))
    return [resp.text for resp in responses]


def main(client, tokenizer, data_name, args):
    samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(samples))
    if len(samples) > 0:
        print(samples[0]["prompt"])

    # start inference
    start_time = time.time()
    prompts = [sample["prompt"] for sample in samples]
    outputs = get_responses(args, client, tokenizer, prompts)
    time_use = time.time() - start_time

    all_samples = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        sample.pop("prompt")
        sample["output"] = output
        sample["generator"] = args.llm_name_or_path.split("/")[-1]
        all_samples.append(sample)

    # Save to a file
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)