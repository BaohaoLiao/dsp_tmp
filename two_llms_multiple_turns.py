import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer
from openai import OpenAI

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--llm1_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--llm1_ip_address", default=None, type=str)
    parser.add_argument("--llm2_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--llm2_ip_address", default=None, type=str)
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
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    openai_api_key = "EMPTY"
    client1 = OpenAI(
        api_key=openai_api_key,
        base_url=args.llm1_ip_address,
    )
    tokenizer1 = AutoTokenizer.from_pretrained(args.llm1_name_or_path, trust_remote_code=True)

    client2 = OpenAI(
        api_key=openai_api_key,
        base_url=args.llm2_ip_address,
    )
    tokenizer2 = AutoTokenizer.from_pretrained(args.llm2_name_or_path, trust_remote_code=True)

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(client1, client2, tokenizer1, tokenizer2, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def get_responses(args, client1, client2, tokenizer1, tokenizer2, prompts):
    outputs = [None] * len(prompts)  # Initialize with None for tracking
    token_counts = [(0, 0) for _ in prompts]  # (client1_tokens, client2_tokens) for each prompt
    turn_info = [[] for _ in prompts]  # List to store (turn_num, client_id) for each prompt
    current_prompts = [(i, p, []) for i, p in enumerate(prompts)] # (index, prompt, responses)
    num_turn = 0
   
    while current_prompts:
        client = client1 if num_turn % 2 == 0 else client2
        tokenizer = tokenizer1 if num_turn % 2 == 0 else tokenizer2
        client_id = 1 if num_turn % 2 == 0 else 2

        batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in current_prompts]
        responses = client.completions.create(
            model=args.llm1_name_or_path.split("/")[-1] if client == client1 else args.llm2_name_or_path.split("/")[-1],
            prompt=batch_prompts,
            temperature=args.temperature,
            top_p=args.top_p, 
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=[args.step_word],
        ).choices
        responses = sorted(responses, key=lambda x: int(x.index))
       
        next_prompts = []
        for (orig_idx, prompt, prev_responses), new_response in zip(current_prompts, responses):
            response_text = new_response.text + args.step_word # response.text doesn't contain step_word
            num_tokens = len(tokenizer.encode(response_text))

            # Update token counts
            if client_id == 1:
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens, token_counts[orig_idx][1])
            else:
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1] + num_tokens)

            # Record turn information
            turn_info[orig_idx].append((num_turn, client_id))

            full_responses = prev_responses + [(response_text, client_id)]
            full_responses_text = ''.join(r[0] for r in full_responses)
            # terminate conditions
            if (new_response.stop_reason is None) \
              or len(tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call:
                outputs[orig_idx] = full_responses_text[:-len(args.step_word)]
            else:
                next_prompts.append((orig_idx, prompt, full_responses)) 
               
        current_prompts = next_prompts
        print(f"Turn {num_turn}: Complete {len(outputs) - len(current_prompts)} / {len(outputs)}")
        num_turn += 1

    # Calculate overall ratio
    total_client1 = sum(counts[0] for counts in token_counts)
    total_client2 = sum(counts[1] for counts in token_counts)
    total_tokens = total_client1 + total_client2
    overall_ratio = (total_client1/total_tokens, total_client2/total_tokens) if total_tokens > 0 else (0,0)
    print(f"Token ratio (client1, client2): {overall_ratio}")

    return outputs, token_counts, turn_info


def main(client1, client2, tokenizer1, tokenizer2, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer1.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        outputs, token_counts, turn_info = get_responses(
            args,
            client1, 
            client2,
            tokenizer1, 
            tokenizer2, 
            prompts, 
        )
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update(
            {"code": code, "pred": preds, "report": reports, 
             "token_counts": token_counts[i], "turn_info": turn_info[i]}
        )
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)