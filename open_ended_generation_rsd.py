import os
import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from external.qwen25_math_evaluation.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--draft_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--target_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--target_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--rm_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
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
    draft_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.draft_ip_address,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_name_or_path, trust_remote_code=True)

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_ip_address,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_name_or_path, trust_remote_code=True)

    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_name_or_path)

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        main(draft_client, draft_tokenizer, target_client, target_tokenizer, rm, rm_tokenizer, data_name, args)

@torch.no_grad()
def prm_scores(prm, prm_tokenizer, current_prompts, responses):
    full_responses = [
        p + "\n\n".join(r[0] for r in prev_resp) + "\n\n" + new_resp.text + "<|eot_id|>"
        for (_, p, prev_resp), new_resp in zip(current_prompts, responses)
    ]
    tok_full_responses = [
        prm_tokenizer(full_response, add_special_tokens=False, return_tensors="pt")['input_ids'].to("cuda:0") for full_response in full_responses
    ]
    all_rewards = []
    for tok_full_response in tok_full_responses:
        all_rewards.append([float(prm(tok_full_response).logits[0][0].item())])
    return all_rewards


def get_responses(args, draft_client, draft_tokenizer, target_client, target_tokenizer, rm, rm_tokenizer, prompts):
    outputs = [None] * len(prompts)  # Initialize with None for tracking
    token_counts = [(0, 0, 0) for _ in prompts]  # (client1_tokens, client2_tokens, discarded_client1_tokens) for each prompt
    turn_info = [[] for _ in prompts]  # List to store (turn_num, client_id) for each prompt
    current_prompts = [(i, p, []) for i, p in enumerate(prompts)] # (index, prompt, responses)
    draft_rewards = [[] for _ in prompts]  
    target_rewards = [[] for _ in prompts] 
    draft_responses = [[] for _ in prompts]  
    target_responses = [[] for _ in prompts]  
    num_turn = 0
    pre_num_finished = 0
    num_unchanged = 0
   
    while current_prompts:
        prm_threshold = args.max_prm_threshold - (args.max_prm_threshold - args.min_prm_threshold) * num_turn / args.max_turns
        batch_prompts = [p + args.step_word.join(r[0] for r in responses) + args.step_word for _, p, responses in current_prompts]

        responses1 = draft_client.completions.create(
            model=args.draft_name_or_path.split("/")[-1],
            prompt=batch_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=[args.step_word],
        ).choices
        responses1 = sorted(responses1, key=lambda x: int(x.index))
        
        # record
        for (orig_idx, _, _), response1 in zip(current_prompts, responses1):
            draft_responses[orig_idx].append(response1.text)

        # Evaluate responses from client1 with PRM
        step_rewards = prm_scores(rm, rm_tokenizer, current_prompts, responses1)

        # Split prompts based on step_reward
        good_prompts = []
        bad_prompts = []
        for (orig_idx, prompt, prev_responses), response1, step_reward in zip(current_prompts, responses1, step_rewards):
            draft_rewards[orig_idx].append(round(step_reward[-1], 4))
            if step_reward[-1] >= prm_threshold:
                good_prompts.append((orig_idx, prompt, prev_responses, response1, True))  # True means use client1
            else:
                response1_text = response1.text + args.step_word
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1], token_counts[orig_idx][2]+len(draft_tokenizer.encode(response1_text)))
                bad_prompts.append((orig_idx, prompt, prev_responses))

        # Generate responses using client2 for bad prompts
        if bad_prompts:
            batch_prompts = [p + args.step_word.join(r[0] for r in responses) + args.step_word for _, p, responses in bad_prompts]
            responses2 = target_client.completions.create(
                model=args.target_name_or_path.split("/")[-1],
                prompt=batch_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call, # - current_max_token_count,
                n=1,
                stop=[args.step_word],
            ).choices
            responses2 = sorted(responses2, key=lambda x: int(x.index))
            
            # Add client2 responses to good_prompts
            for (orig_idx, prompt, prev_responses), response2 in zip(bad_prompts, responses2):
                good_prompts.append((orig_idx, prompt, prev_responses, response2, False))  # False means use client2
                # record
                target_responses[orig_idx].append(response2.text)

            # Evaluate responses from client2 with PRM, only record for analysis
            step_rewards = prm_scores(rm, rm_tokenizer, bad_prompts, responses2)

            for i, (orig_idx, _, _) in enumerate(bad_prompts):
                target_rewards[orig_idx].append(round(step_rewards[i][-1], 4))

        # Process all responses
        next_prompts = []
        next_problems = []
        for orig_idx, prompt, prev_responses, response, used_client1 in sorted(good_prompts, key=lambda x: x[0]):
            response_text = response.text + args.step_word
            client_id = 1 if used_client1 else 2
            tokenizer = draft_tokenizer if client_id == 1 else target_tokenizer
            num_tokens = len(tokenizer.encode(response_text))
            
            # Update token counts
            if client_id == 1:
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens, token_counts[orig_idx][1], token_counts[orig_idx][2])
            else:
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1] + num_tokens, token_counts[orig_idx][2])
            
            # Record turn information
            turn_info[orig_idx].append((num_turn, client_id))

            full_responses = prev_responses + [(response.text, client_id)]
            full_responses_text = args.step_word.join(r[0] for r in full_responses) + args.step_word
            
            # terminate conditions
            if (response.stop_reason is None) \
             or len(draft_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
             or len(target_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
             or num_turn >= args.max_turns - 1 \
             or num_unchanged >= args.patience - 1:
                outputs[orig_idx] = full_responses_text[:-len(args.step_word)]
            else:
                next_prompts.append((orig_idx, prompt, full_responses))
                
        current_prompts = next_prompts
        current_problems = next_problems
        assert len(current_prompts) == len(current_problems)
        if len(outputs) - len(current_prompts) > pre_num_finished:
            num_unchanged = 0
            pre_num_finished = len(outputs) - len(current_prompts)
        else:
            num_unchanged += 1

        print(f"#### Step {num_turn}: Completed {pre_num_finished} / {len(outputs)}, #unchanged {num_unchanged} / {args.patience}")
        num_turn += 1

    return outputs, token_counts, turn_info, draft_rewards, target_rewards, draft_responses, target_responses


def main(draft_client, draft_tokenizer, target_client, target_tokenizer, rm, rm_tokenizer, data_name, args):
    samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(samples))
    if len(samples) > 0:
        print(samples[0]["prompt"])

    # start inference
    start_time = time.time()
    prompts = [sample["prompt"] for sample in samples]
    outputs, token_counts, turn_info, draft_rewards, target_rewards, draft_responses, target_responses = get_responses(args, draft_client, draft_tokenizer, target_client, target_tokenizer, rm, rm_tokenizer, prompts)
    time_use = time.time() - start_time

    all_samples = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        sample.pop("prompt")
        sample["output"] = output
        sample["generator"] = args.llm_name_or_path.split("/")[-1]
        sample.update(
            { 
             "draft_response": draft_responses[i], 
             "target_response": target_responses[i],
             "token_counts": token_counts[i], 
             "turn_info": turn_info[i], 
             "draft_reward": draft_rewards[i],
             "target_reward": target_rewards[i], 
            }
        )
        all_samples.append(sample)

    # Save to a file
    print(f"Save output to {out_file}")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)


    llm1_tokens = [0, 0] # (correct, wrong)
    llm1_discarded_tokens = [0, 0]
    llm2_tokens = [0, 0]
    for i, sample in enumerate(all_samples):
        if sample["score"][0]:
            llm1_tokens[0] += sample["token_counts"][0]
            llm2_tokens[0] += sample["token_counts"][1]
            llm1_discarded_tokens[0] += sample["token_counts"][2]
        else:
            llm1_tokens[1] += sample["token_counts"][0]
            llm2_tokens[1] += sample["token_counts"][1]
            llm1_discarded_tokens[1] += sample["token_counts"][2]
    total_tokens = sum(llm1_tokens) + sum(llm2_tokens) + sum(llm1_discarded_tokens)
    total_tokens_for_correct_pred = llm1_discarded_tokens[0] + llm1_tokens[0] + llm2_tokens[0]
    total_tokens_for_wrong_pred = llm1_discarded_tokens[1] + llm1_tokens[1] + llm2_tokens[1]

    result_json = {}
    result_json["tokens_ratio_overall(llm1,llm2)"] = (
        (sum(llm1_tokens)+sum(llm1_discarded_tokens))/total_tokens, sum(llm2_tokens)/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_correct_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[0]+llm1_tokens[0])/total_tokens_for_correct_pred, llm2_tokens[0]/total_tokens_for_correct_pred
    ) if total_tokens_for_correct_pred > 0 else (0,0) 
    result_json["tokens_ratio_wrong_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[1]+llm1_tokens[1])/total_tokens_for_wrong_pred, llm2_tokens[1]/total_tokens_for_wrong_pred
    ) if total_tokens_for_wrong_pred > 0 else (0,0) 
    result_json["tokens_ratio(correct,wrong)"] = (
        total_tokens_for_correct_pred/total_tokens, total_tokens_for_wrong_pred/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_discarded(correct,wrong)"] = (
        llm1_discarded_tokens[0]/total_tokens_for_correct_pred, llm1_discarded_tokens[1]/total_tokens_for_wrong_pred
    ) if (total_tokens_for_correct_pred > 0 and total_tokens_for_wrong_pred > 0)  else (0,0) 
    result_json["acceptance_rate"] = (
        (llm1_tokens[0] + llm1_tokens[1])/(llm1_tokens[0] + llm1_tokens[1] + llm1_discarded_tokens[0] + llm1_discarded_tokens[1])
    ) if ((llm1_tokens[0] + llm1_tokens[1]) > 0)  else (0,0) 
    result_json["num_draft_tokens"] = sum(llm1_tokens) + sum(llm1_discarded_tokens)
    result_json["num_target_tokens"] = sum(llm2_tokens)

    with open(
        out_file.replace(".json", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)