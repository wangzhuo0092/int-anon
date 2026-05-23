import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils import get_data_name, load_data


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_feedback", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)

    return parser.parse_args()


def softmax(values):
    """Compute softmax over a list of log-probabilities."""
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values) / np.exp(values).sum()
    return probs.tolist()


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


def build_messages(prompt, data_name, turn=None, with_feedback=False):
    """Build API messages consistent with open-source judge prompts."""
    prompt_dict = build_system_prompts()
    mode = "reasoning" if with_feedback else "direct"

    if data_name == "chatbot_arena":
        system_prompt = prompt_dict["chatbot_arena"][mode]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    if data_name == "mt_bench_turn1":
        system_prompt = prompt_dict["mt_bench_turn1"][mode]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    if data_name == "mt_bench_turn2":
        system_prompt = prompt_dict["mt_bench_turn2"][mode]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    if data_name == "mt_bench":
        if turn == 1:
            system_prompt = prompt_dict["mt_bench_turn1"][mode]
        elif turn == 2:
            system_prompt = prompt_dict["mt_bench_turn2"][mode]
        else:
            raise ValueError(f"Unknown MT-Bench turn: {turn}")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    return [{"role": "user", "content": prompt}]


def build_candidate_labels(data_name, points):
    """Build candidate label values for absolute scoring or pairwise preference."""
    if data_name in ["chatbot_arena", "mt_bench_turn1", "mt_bench_turn2", "mt_bench"]:
        return [0, 1, 2]

    return list(range(1, points + 1))


def normalize_token(token):
    """Normalize generated tokens before matching candidate labels."""
    if token is None:
        return ""

    return str(token).strip().strip("\"'`").strip("：:，,。.;；")


def extract_score_logprobs(choice, label_values, missing_logprob=-30.0):
    """
    Extract candidate-label log-probabilities at the detected score-token position.

    The function scans generated tokens from right to left and uses the last
    valid label token as the final score token.
    """
    if choice.logprobs is None or choice.logprobs.content is None:
        return None

    contents = choice.logprobs.content
    label_strings = [str(value) for value in label_values]
    label_set = set(label_strings)

    generated_tokens = [normalize_token(item.token) for item in contents]

    if len(contents) == 0:
        label_logprobs = [missing_logprob] * len(label_values)
        return {
            "success": False,
            "score_pos": None,
            "generated_tokens": generated_tokens,
            "label_logprobs": label_logprobs,
            "probs": softmax(label_logprobs),
            "direct_score": -1,
            "weighted_score": -1,
        }

    score_pos = None
    score_token = None

    for pos in range(len(contents) - 1, -1, -1):
        token = normalize_token(contents[pos].token)
        if token in label_set:
            score_pos = pos
            score_token = token
            break

    if score_pos is None:
        label_logprobs = [missing_logprob] * len(label_values)
        return {
            "success": False,
            "score_pos": None,
            "generated_tokens": generated_tokens,
            "label_logprobs": label_logprobs,
            "probs": softmax(label_logprobs),
            "direct_score": -1,
            "weighted_score": -1,
        }

    top_logprobs = contents[score_pos].top_logprobs or []
    logprob_dict = {
        normalize_token(item.token): float(item.logprob)
        for item in top_logprobs
    }

    label_logprobs = [
        logprob_dict.get(str(value), missing_logprob)
        for value in label_values
    ]
    probs = softmax(label_logprobs)

    direct_score = int(score_token)
    weighted_score = float(
        sum(value * prob for value, prob in zip(label_values, probs))
    )

    return {
        "success": True,
        "score_pos": score_pos,
        "generated_tokens": generated_tokens,
        "label_logprobs": label_logprobs,
        "probs": probs,
        "direct_score": direct_score,
        "weighted_score": weighted_score,
    }


def call_api(client, model_name, messages, args):
    """Call the chat completion API with token log-probability extraction."""
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1,
        seed=args.seed,
        logprobs=True,
        top_logprobs=args.top_logprobs,
    )


def build_output_record(idx, prompt, human_score, response_text, parsed, model_name, top_logprobs):
    """Build a JSON-serializable output record for one sample."""
    df = pd.DataFrame([
        {
            "layer_n": "api_final",
            "direct_score": parsed["direct_score"],
            "weighted_score": parsed["weighted_score"],
            "probs": parsed["probs"],
            "logits": parsed["label_logprobs"],
            "label_logprobs": parsed["label_logprobs"],
            "score_pos": parsed["score_pos"],
            "success": parsed["success"],
        }
    ])

    direct_score = float(parsed["direct_score"]) if parsed["direct_score"] != -1 else -1
    weighted_score = float(parsed["weighted_score"]) if parsed["weighted_score"] != -1 else -1

    return {
        "idx": idx,
        "prompt": prompt,
        "human_score": human_score,
        "response": response_text,
        "direct_socre": direct_score,
        "weighted_socre": weighted_score,
        "weighted_direct_socre": direct_score,
        "df": df.to_dict(),
        "api_meta": {
            "model": model_name,
            "top_logprobs": top_logprobs,
            "is_real_logits": False,
            "note": (
                "The logits field stores candidate-label top log-probabilities "
                "at the detected score-token position, not full-vocabulary logits."
            ),
            "generated_tokens": parsed["generated_tokens"],
        },
    }


def build_error_record(idx, prompt, human_score, model_name, top_logprobs, error):
    """Build a fallback record when API generation or parsing fails."""
    return {
        "idx": idx,
        "prompt": prompt,
        "human_score": human_score,
        "response": "",
        "direct_socre": -1,
        "weighted_socre": -1,
        "weighted_direct_socre": -1,
        "df": {},
        "api_meta": {
            "model": model_name,
            "top_logprobs": top_logprobs,
            "is_real_logits": False,
            "error": repr(error),
        },
    }


def load_api_key(args):
    """Load API key from arguments or environment variables."""
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

    if api_key is None:
        raise ValueError(
            "Please provide --api_key or set OPENAI_API_KEY / OPENROUTER_API_KEY."
        )

    return api_key


def main():
    args = get_args()

    if args.top_logprobs <= 0:
        raise ValueError("--top_logprobs must be greater than 0.")

    api_key = load_api_key(args)
    client = (
        OpenAI(api_key=api_key, base_url=args.base_url)
        if args.base_url
        else OpenAI(api_key=api_key)
    )

    data_name = get_data_name(args.input_file)
    safe_model_name = args.model_name.replace("/", "__").replace(":", "_")

    output_dir = os.path.join(args.save_dir, data_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{safe_model_name}{'_with_feedback' if args.with_feedback else ''}_api_logprobs.json",
    )

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skip")
        return

    if data_name == "mt_bench":
        human_scores, all_prompts, turn_list = load_data(
            args.input_file,
            args.with_feedback,
        )
    else:
        human_scores, all_prompts = load_data(args.input_file, args.with_feedback)
        turn_list = [None] * len(all_prompts)

    label_values = build_candidate_labels(data_name, args.points)

    print("data_name =", data_name)
    print("model_name =", args.model_name)
    print("label_values =", label_values)
    print("len(human_scores) =", len(human_scores))
    print("len(all_prompts) =", len(all_prompts))
    print("output_path =", output_path)

    processed_results = []

    iterator = zip(all_prompts, human_scores, turn_list)
    for idx, (prompt, human_score, turn) in enumerate(
        tqdm(iterator, total=len(all_prompts))
    ):
        messages = build_messages(
            prompt=prompt,
            data_name=data_name,
            turn=turn,
            with_feedback=args.with_feedback,
        )

        try:
            response = call_api(
                client=client,
                model_name=args.model_name,
                messages=messages,
                args=args,
            )

            choice = response.choices[0]
            response_text = choice.message.content or ""

            parsed = extract_score_logprobs(
                choice=choice,
                label_values=label_values,
            )

            if parsed is None:
                raise ValueError(
                    "No logprobs returned. Please check whether this model/API supports logprobs."
                )

            item = build_output_record(
                idx=idx,
                prompt=prompt,
                human_score=human_score,
                response_text=response_text,
                parsed=parsed,
                model_name=args.model_name,
                top_logprobs=args.top_logprobs,
            )

        except Exception as error:
            item = build_error_record(
                idx=idx,
                prompt=prompt,
                human_score=human_score,
                model_name=args.model_name,
                top_logprobs=args.top_logprobs,
                error=error,
            )

        processed_results.append(item)

        if args.sleep > 0:
            time.sleep(args.sleep)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)

    print("saved:", output_path)
    print("len(processed_results) =", len(processed_results))


if __name__ == "__main__":
    main()