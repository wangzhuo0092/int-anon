import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import os
import warnings

import torch

from utils import (
    get_batch_inputs,
    get_layer_outputs,
    get_data_name,
    load_data,
    load_model_and_tokenizer,
)

warnings.filterwarnings("ignore")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--points", type=int, default=5, help="Maximum score value.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--with_feedback", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)

    return parser.parse_args()


def build_candidate_labels(data_name, points):
    """Build candidate label values for absolute scoring or pairwise preference."""
    if data_name in ["chatbot_arena", "mt_bench_turn1", "mt_bench_turn2", "mt_bench"]:
        return [0, 1, 2]

    return list(range(1, points + 1))


def build_system_prompts():
    """Return system prompts for pairwise preference benchmarks."""
    chatbot_arena_direct = (
        "You are a highly efficient assistant, who evaluates and rank large language models "
        "(LLMs) based on the quality of their responses to given prompts. This process will "
        "create a leaderboard reflecting the most accurate and human-preferred answers."
    )

    chatbot_arena_reasoning = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. The input prompt can possibly include an image; in that case the user question or instruction will be related to that image and you must take that into account. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Compare assistant A's answer and assistant B's answer directly. Identify any mistakes, inaccuracies, omissions, or weaknesses in the two responses.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

0
1
2

Use:
- 0 if Assistant A's Answer is better
- 1 if the two Answers are equally good or equally bad and result in a tie
- 2 if Assistant B's Answer is better

Do not output A, B, tie, Choice, label, punctuation, or any explanation after the final verdict.
Your final verdict must be exactly one of: 0, 1, 2.

Example output: "Explanation: <your brief explanation here>. Final verdict: 0"."""

    mt_bench_turn1_direct = (
        "Please act as an impartial judge and evaluate the quality of the responses provided "
        "by two AI assistants to the user question displayed below. Your evaluation should "
        "consider correctness and helpfulness."
    )

    mt_bench_turn1_reasoning = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.

You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the responses. Begin your evaluation by comparing the two responses and provide a short explanation.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following one of these three formats:

0
1
2

Use:
- 0 if Assistant A's Answer is better
- 1 if the two answers are equally good or equally bad
- 2 if Assistant B's Answer is better

Do not output A, B, tie, Choice, label, punctuation, or any explanation after the final verdict.
Your final verdict must be exactly one of: 0, 1, 2.

Example output: "Explanation: <your brief explanation here>. Final verdict: 0"."""

    mt_bench_turn2_direct = (
        "Please act as an impartial judge and evaluate the quality of the responses provided "
        "by two AI assistants in a multi-turn conversation displayed below. Focus on which "
        "assistant gives the better answer to the SECOND user question, considering each "
        "conversation's previous context."
    )

    mt_bench_turn2_reasoning = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants in a multi-turn conversation displayed below.

You should choose the assistant that gives a better answer to the SECOND user question, while taking each assistant's previous conversation context into account. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, level of detail, and consistency with the previous context. Begin your evaluation by comparing the two conversations and provide a short explanation.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following one of these three formats:

0
1
2

Use:
- 0 if Assistant A is better on the second user question
- 1 if the two answers are equally good or equally bad on the second user question
- 2 if Assistant B is better on the second user question

Do not output A, B, tie, Choice, label, punctuation, or any explanation after the final verdict.
Your final verdict must be exactly one of: 0, 1, 2.

Example output: "Explanation: <your brief explanation here>. Final verdict: 0"."""

    return {
        "chatbot_arena": {
            "direct": chatbot_arena_direct,
            "reasoning": chatbot_arena_reasoning,
        },
        "mt_bench_turn1": {
            "direct": mt_bench_turn1_direct,
            "reasoning": mt_bench_turn1_reasoning,
        },
        "mt_bench_turn2": {
            "direct": mt_bench_turn2_direct,
            "reasoning": mt_bench_turn2_reasoning,
        },
    }


def apply_chat_template(tokenizer, prompt, system_prompt=None):
    """Apply the model chat template to a single prompt."""
    if system_prompt is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_tokenized_inputs(args, data_name, tokenizer, model, all_prompts, turn_list=None):
    """Build tokenized inputs for generation and layer-wise logit extraction."""
    non_batched_datasets = [
        "biggen",
        "chatbot_arena",
        "mt_bench_turn1",
        "mt_bench_turn2",
        "mt_bench",
    ]

    if data_name not in non_batched_datasets:
        tokenized_inputs, save_idx_list = get_batch_inputs(
            all_prompts,
            tokenizer,
            batch_size=args.batch_size,
            device=f"cuda:{torch.cuda.device_count() - 1}",
            max_length=args.max_length,
        )
        return tokenized_inputs, save_idx_list

    prompt_dict = build_system_prompts()
    use_reasoning = "reasoning" if args.with_feedback else "direct"

    all_text = []

    if data_name == "chatbot_arena":
        system_prompt = prompt_dict["chatbot_arena"][use_reasoning]
        all_text = [
            apply_chat_template(tokenizer, prompt, system_prompt)
            for prompt in all_prompts
        ]

    elif data_name in ["mt_bench_turn1", "mt_bench_turn2"]:
        system_prompt = prompt_dict[data_name][use_reasoning]
        all_text = [
            apply_chat_template(tokenizer, prompt, system_prompt)
            for prompt in all_prompts
        ]

    elif data_name == "mt_bench":
        for prompt, turn in zip(all_prompts, turn_list):
            if turn == 1:
                system_prompt = prompt_dict["mt_bench_turn1"][use_reasoning]
            elif turn == 2:
                system_prompt = prompt_dict["mt_bench_turn2"][use_reasoning]
            else:
                raise ValueError(f"Unknown MT-Bench turn: {turn}")

            all_text.append(apply_chat_template(tokenizer, prompt, system_prompt))

    else:
        all_text = [
            apply_chat_template(tokenizer, prompt, system_prompt=None)
            for prompt in all_prompts
        ]

    print("len(all_text) =", len(all_text))

    tokenized_inputs = []
    for i, text in enumerate(all_text):
        batch_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        tokenized_inputs.append(
            {
                "batch_prompts": [all_prompts[i]],
                "batch_inputs": batch_inputs,
                "text_batch": text,
            }
        )

    save_idx_list = list(range(len(all_prompts)))
    return tokenized_inputs, save_idx_list


def build_output_record(result, idx, human_score):
    """Convert one generated result into a JSON-serializable record."""
    prompt = result["prompt"]
    layer_df = result["res"]

    raw_argmax_score = layer_df.iloc[-1]["direct_score"]
    raw_expected_score = layer_df.iloc[-1]["weighted_score"]
    mean_layer_argmax_score = layer_df["direct_score"].mean().item()

    return {
        "idx": idx,
        "prompt": prompt,
        "human_score": human_score,
        "direct_socre": float(raw_argmax_score),
        "weighted_socre": float(raw_expected_score),
        "weighted_direct_socre": float(mean_layer_argmax_score),
        "df": layer_df.to_dict(),
    }


def main():
    args = get_args()

    data_name = get_data_name(args.input_file)
    model_name = args.model_name_or_path.split("/")[-1]

    output_dir = os.path.join(args.save_dir, data_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{model_name}{'_with_feedback' if args.with_feedback else ''}_logits.json",
    )

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skip")
        return

    print(f"{output_path} does not exist, start generation")

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    label_values = build_candidate_labels(data_name, args.points)
    label_token_ids = [
        tokenizer.convert_tokens_to_ids([str(label_value)])[0]
        for label_value in label_values
    ]

    if data_name == "mt_bench":
        human_scores, all_prompts, turn_list = load_data(args.input_file, args.with_feedback)
    else:
        human_scores, all_prompts = load_data(args.input_file, args.with_feedback)
        turn_list = None

    print("len(human_scores) =", len(human_scores))
    print("len(all_prompts) =", len(all_prompts))

    tokenized_inputs, save_idx_list = build_tokenized_inputs(
        args=args,
        data_name=data_name,
        tokenizer=tokenizer,
        model=model,
        all_prompts=all_prompts,
        turn_list=turn_list,
    )

    all_results = get_layer_outputs(
        model,
        tokenized_inputs,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        points_ids_list=label_token_ids,
        label_values=label_values,
        temperature=args.temperature,
    )

    print("len(all_results) =", len(all_results))
    print("len(save_idx_list) =", len(save_idx_list))

    processed_results = []

    for i, result in enumerate(all_results):
        idx = save_idx_list[i]
        human_score = human_scores[idx]

        record = build_output_record(
            result=result,
            idx=idx,
            human_score=human_score,
        )
        processed_results.append(record)

    print("len(processed_results) =", len(processed_results))
    print("processed idxs =", [x["idx"] for x in processed_results])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=4)


if __name__ == "__main__":
    main()