import datetime
import json
import logging
import warnings
from logging.handlers import RotatingFileHandler

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import colorlog
except ImportError:
    colorlog = None


def load_model_and_tokenizer(model_path, dtype=torch.bfloat16):
    """Load a causal language model and its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def validate_model_and_data_consistency(model_path, valid_data_path):
    """Check whether the model path and validation file refer to the same model."""
    supported_models = [
        "mistral-7b-instruct-v0.3",
        "internlm3-8b-instruct",
        "llama-3___1-8b-instruct",
        "qwen-2___5-14b-instruct",
        "mistral-small-24b-instruct",
        "llama-3___3-70b-instruct",
    ]

    model_path_lower = model_path.lower()
    valid_data_path_lower = valid_data_path.lower()

    is_consistent = any(
        model_key in model_path_lower and model_key in valid_data_path_lower
        for model_key in supported_models
    )

    if not is_consistent:
        raise ValueError(
            "model_path and valid_data_path should refer to the same model."
        )


def validate_data_consistency(data_path, valid_data_path):
    """Return the matched model key if two result paths refer to the same model."""
    supported_models = [
        "mistral-7b-instruct-v0___3",
        "internlm3-8b-instruct",
        "llama-3___1-8b-instruct",
        "qwen-2___5-14b-instruct",
        "mistral-small-24b-instruct",
        "llama-3___3-70b-instruct",
    ]

    data_path_lower = data_path.lower()
    valid_data_path_lower = valid_data_path.lower()

    for model_key in supported_models:
        if model_key in data_path_lower and model_key in valid_data_path_lower:
            return model_key

    raise ValueError("data_path and valid_data_path should refer to the same model.")


def get_data_name(data_path):
    """Infer the dataset name from a file path."""
    path = data_path.lower()

    if "mt_bench_turn1" in path:
        return "mt_bench_turn1"
    if "mt_bench_turn2" in path:
        return "mt_bench_turn2"
    if "mt_bench" in path:
        return "mt_bench"
    if "chatbot_arena" in path:
        return "chatbot_arena"
    if "flask" in path:
        return "flask"
    if "valid" in path:
        return "valid"
    if "helpsteer" in path:
        return "helpsteer"
    if "biggen" in path:
        return "biggen"

    return data_path.split("/")[-1].split(".")[0]


def load_data(data_path, with_feedback=False):
    """
    Load prompts and human annotations from a JSONL data file.

    For MT-Bench, this function also returns the turn index.
    """
    scores = []
    prompts = []
    turns = []

    data_name = get_data_name(data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            scores.append(sample["score"])
            prompts.append(
                sample["user_prompt_feedback"]
                if with_feedback
                else sample["user_prompt"]
            )

            if data_name == "mt_bench":
                turns.append(int(sample["turn"]))

            if data_name == "helpsteer" and len(prompts) > 2000:
                break

    if data_name == "mt_bench":
        return scores, prompts, turns

    return scores, prompts


def apply_chat_template(tokenizer, messages):
    """Apply the chat template with a safe fallback for tokenizer-specific options."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_batch_inputs(prompts, tokenizer, batch_size=8, device="cuda", max_length=2048):
    """
    Convert prompts into batched tokenized inputs.

    Returns:
        tokenized_inputs: A list of batch dictionaries.
        save_idx_list: Original indices kept after length filtering.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    all_text = [
        apply_chat_template(tokenizer, [{"role": "user", "content": prompt}])
        for prompt in prompts
    ]

    encoded = tokenizer(all_text, add_special_tokens=False, truncation=False)

    kept_items = [
        (idx, text, len(input_ids))
        for idx, (text, input_ids) in enumerate(zip(all_text, encoded["input_ids"]))
        if len(input_ids) <= max_length
    ]

    kept_items.sort(key=lambda item: item[2])

    tokenized_inputs = []
    save_idx_list = [item[0] for item in kept_items]

    for start in range(0, len(kept_items), batch_size):
        chunk = kept_items[start:start + batch_size]

        text_batch = [item[1] for item in chunk]
        idx_batch = [item[0] for item in chunk]
        batch_prompts = [prompts[idx] for idx in idx_batch]

        batch_inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        tokenized_inputs.append(
            {
                "batch_prompts": batch_prompts,
                "batch_inputs": batch_inputs,
                "text_batch": text_batch,
            }
        )

    return tokenized_inputs, save_idx_list


def get_layer_outputs(
    model,
    tokenized_inputs,
    tokenizer,
    max_new_tokens,
    points_ids_list,
    label_values,
    temperature=0,
):
    """
    Generate judge outputs and extract layer-wise candidate-label logits.

    Args:
        model: The judge model.
        tokenized_inputs: Batched tokenized inputs.
        tokenizer: The tokenizer.
        max_new_tokens: Maximum number of generated tokens.
        points_ids_list: Token ids of candidate labels.
        label_values: Candidate labels, e.g., [1, 2, 3, 4, 5] or [0, 1, 2].
        temperature: Decoding temperature.

    Returns:
        A list of dictionaries. Each item contains the original prompt and a
        DataFrame with layer-wise direct scores, expected scores, probabilities,
        and logits.
    """
    all_results = []
    debug_count = 0

    print("label_values:", label_values)
    print("points_ids_list:", points_ids_list)

    for token_id, label in zip(points_ids_list, label_values):
        print(label, token_id, repr(tokenizer.decode([token_id])))

    do_sample = temperature != 0

    for block_inputs in tqdm(tokenized_inputs):
        prompts = block_inputs["batch_prompts"]
        batch_inputs = block_inputs["batch_inputs"]

        outputs = model.generate(
            **batch_inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
            temperature=temperature,
            top_k=5,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )

        prompt_length = batch_inputs["input_ids"].shape[-1]
        response_ids = outputs.sequences[:, prompt_length:]

        columns = ["layer_n", "direct_score", "weighted_score", "probs", "logits"]

        for batch_idx in range(response_ids.shape[0]):
            score_pos = None

            for pos in range(response_ids.shape[1] - 1, -1, -1):
                if response_ids[batch_idx, pos].item() in points_ids_list:
                    score_pos = pos
                    break

            if debug_count < 4:
                print("response_ids:", response_ids[batch_idx].tolist())
                print(
                    "response_tokens:",
                    [repr(tokenizer.decode([x])) for x in response_ids[batch_idx].tolist()],
                )
                print(
                    "decoded_response:",
                    tokenizer.decode(response_ids[batch_idx], skip_special_tokens=False),
                )
                print("score_pos:", score_pos)
                debug_count += 1

            if score_pos is None:
                invalid_result = pd.DataFrame(
                    [[-1, -1, -1, [-1] * len(label_values), [-1] * len(label_values)]],
                    columns=columns,
                )
                all_results.append({"prompt": prompts[batch_idx], "res": invalid_result})
                continue

            score_hidden_states = outputs.hidden_states[score_pos]
            rows = []

            for layer_idx, layer_hidden_state in enumerate(score_hidden_states):
                last_token_hidden = layer_hidden_state[batch_idx, -1, :]

                logits = model.lm_head(last_token_hidden)
                label_logits = logits[points_ids_list].to(torch.float32)

                probs = torch.softmax(label_logits, dim=-1)
                probs_cpu = probs.detach().cpu().tolist()
                logits_cpu = label_logits.detach().cpu().tolist()

                direct_score = label_values[probs.argmax(dim=-1).item()]
                weighted_score = sum(
                    label * prob for label, prob in zip(label_values, probs_cpu)
                )

                rows.append(
                    {
                        "layer_n": layer_idx,
                        "direct_score": direct_score,
                        "weighted_score": weighted_score,
                        "probs": probs_cpu,
                        "logits": logits_cpu,
                    }
                )

            result_df = pd.DataFrame(rows, columns=columns)
            all_results.append({"prompt": prompts[batch_idx], "res": result_df})

    return all_results


def setup_logger(name, log_file="output.log", level=logging.DEBUG):
    """Create a file-and-console logger."""
    def utc_plus_8(sec, what):
        local_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return local_time.timetuple()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.Formatter.converter = utc_plus_8

    if logger.hasHandlers():
        return logger

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=1,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(
        "%(asctime)s[%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if colorlog is not None:
        console_format = (
            "%(log_color)s%(asctime)s-%(threadName)s-%(filename)s-"
            "[line:%(lineno)d]-%(levelname)s: %(message)s"
        )
        color_config = {
            "DEBUG": "white",
            "INFO": "white",
            "WARNING": "blue",
            "ERROR": "yellow",
            "CRITICAL": "red",
        }
        console_formatter = colorlog.ColoredFormatter(
            fmt=console_format,
            log_colors=color_config,
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s-%(threadName)s-%(filename)s-"
            "[line:%(lineno)d]-%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logger configured.")
    return logger


def find_sublist_position(main_list, sublist):
    """Return the start position of a sublist, or -1 if it is not found."""
    length = len(sublist)

    for idx in range(len(main_list) - length + 1):
        if main_list[idx:idx + length] == sublist:
            return idx

    return -1


def align_layer_weights(layer_logits, weights):
    """Align layer weights with extracted layer logits."""
    weights = torch.as_tensor(weights, dtype=torch.float32)

    if layer_logits.size(0) == weights.numel():
        return layer_logits, weights

    if layer_logits.size(0) == weights.numel() + 1:
        return layer_logits[:-1], weights

    raise ValueError(
        f"Layer mismatch: logits={layer_logits.size(0)}, weights={weights.numel()}"
    )


def get_score(
    model,
    tokenized_inputs,
    tokenizer,
    weights,
    temperature=0,
    max_new_tokens=16,
    max_score=5,
    score_token="score",
):
    """
    Legacy helper for extracting a single final score and internal scores.

    This function is kept for compatibility with earlier small-scale scoring
    scripts. The main pipeline should use get_layer_outputs instead.
    """
    if max_score == 5:
        default_score = 3
    elif max_score == 9:
        default_score = 5
    else:
        default_score = max_score // 2

    default_return = (
        default_score,
        default_score,
        default_score,
        default_score,
        None,
    )

    if len(tokenized_inputs) == 0:
        return default_return

    try:
        points_ids_list = [
            tokenizer.convert_tokens_to_ids([str(i)])[0]
            for i in range(1, max_score + 1)
        ]
    except Exception:
        points_ids_list = [
            tokenizer.convert_tokens_to_ids([str(i).encode()])[0]
            for i in range(1, max_score + 1)
        ]

    label_values = list(range(1, max_score + 1))
    columns = ["layer_n", "direct_score", "weighted_score", "logits", "probs"]

    do_sample = temperature != 0

    for block_inputs in tokenized_inputs:
        prompts = block_inputs["batch_prompts"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            outputs = model.generate(
                **block_inputs["batch_inputs"],
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        response_ids = outputs.sequences[
            :,
            block_inputs["batch_inputs"]["input_ids"].shape[-1]:,
        ]

        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        print(responses[0])

        output_ids = response_ids[0].tolist()

        if score_token == "score":
            score_ids = [
                tokenizer.encode("score:")[-2:],
                tokenizer.encode(" score:")[-2:],
                tokenizer.encode("\nscore:")[-2:],
                tokenizer.encode("\n\nscore:")[-2:],
                tokenizer.encode("score: ")[-2:],
                tokenizer.encode(" score: ")[-2:],
                tokenizer.encode("\nscore: ")[-2:],
                tokenizer.encode("\n\nscore: ")[-2:],
                tokenizer.encode("Score:")[-2:],
                tokenizer.encode(" Score:")[-2:],
                tokenizer.encode("\nScore:")[-2:],
                tokenizer.encode("\n\nScore:")[-2:],
                tokenizer.encode("Score: ")[-2:],
                tokenizer.encode(" Score: ")[-2:],
                tokenizer.encode("\nScore: ")[-2:],
                tokenizer.encode("\n\nScore: ")[-2:],
            ]
        elif score_token == "level":
            score_ids = [
                tokenizer.encode("level:")[-2:],
                tokenizer.encode(" level:")[-2:],
                tokenizer.encode("\nlevel:")[-2:],
                tokenizer.encode("\n\nlevel:")[-2:],
                tokenizer.encode("level: ")[-2:],
                tokenizer.encode(" level: ")[-2:],
                tokenizer.encode("\nlevel: ")[-2:],
                tokenizer.encode("\n\nlevel: ")[-2:],
                tokenizer.encode("Level:")[-2:],
                tokenizer.encode(" Level:")[-2:],
                tokenizer.encode("\nLevel:")[-2:],
                tokenizer.encode("\n\nLevel:")[-2:],
                tokenizer.encode("Level: ")[-2:],
                tokenizer.encode(" Level: ")[-2:],
                tokenizer.encode("\nLevel: ")[-2:],
                tokenizer.encode("\n\nLevel: ")[-2:],
            ]
        else:
            raise ValueError(f"Unsupported score_token: {score_token}")

        score_pos = -1
        for score_id in score_ids:
            score_pos = find_sublist_position(output_ids, score_id)
            if score_pos != -1:
                break

        if score_pos == -1:
            return default_return

        candidate_positions = []
        for relative_pos, token_id in enumerate(output_ids[score_pos:]):
            if tokenizer.decode(token_id) in [str(i) for i in range(1, max_score + 1)]:
                candidate_positions.append(relative_pos)

        if len(candidate_positions) == 0:
            print(responses[0])
            return default_return

        score_generation_pos = score_pos + candidate_positions[0]
        score_hidden_states = outputs.hidden_states[score_generation_pos]

        rows = []

        for layer_idx, layer_hidden_state in enumerate(score_hidden_states):
            last_token_hidden = layer_hidden_state[0, -1, :]
            logits = model.lm_head(last_token_hidden)

            label_logits = logits[points_ids_list].to(torch.float32)
            probs = torch.softmax(label_logits, dim=-1)

            direct_score = label_values[probs.argmax(dim=-1).item()]
            probs_cpu = probs.detach().cpu().tolist()
            logits_cpu = label_logits.detach().cpu().tolist()

            weighted_score = sum(
                label * prob for label, prob in zip(label_values, probs_cpu)
            )

            rows.append(
                {
                    "layer_n": layer_idx,
                    "direct_score": direct_score,
                    "weighted_score": weighted_score,
                    "logits": logits_cpu,
                    "probs": probs_cpu,
                }
            )

        result_df = pd.DataFrame(rows, columns=columns)

        direct_score = str(result_df["direct_score"].tolist()[-1])
        weighted_score = str(result_df["weighted_score"].tolist()[-1])

        layer_logits = torch.stack(
            [
                torch.tensor(logits, dtype=torch.float32)
                for logits in result_df["logits"].tolist()
            ],
            dim=0,
        )

        aligned_logits, aligned_weights = align_layer_weights(layer_logits, weights)
        score_values = torch.arange(1, max_score + 1, dtype=torch.float32)

        int_logit_w_probs = torch.softmax(
            (aligned_logits * aligned_weights.view(-1, 1)).sum(dim=0),
            dim=-1,
        )
        int_logit_w_score = (int_logit_w_probs * score_values).sum().item()

        int_logit_avg_probs = torch.softmax(aligned_logits.mean(dim=0), dim=-1)
        int_logit_avg_score = (int_logit_avg_probs * score_values).sum().item()

        return (
            direct_score,
            weighted_score,
            int_logit_w_score,
            int_logit_avg_score,
            result_df,
        )

    return default_return


def get_final_score(full_prompt, model, tokenizer, weights, max_new_tokens, max_score):
    """Run the legacy single-prompt scoring helper."""
    prompts = [full_prompt]

    tokenized_inputs = get_batch_inputs(
        prompts,
        tokenizer,
        batch_size=1,
    )[0]

    direct_score, weighted_score, int_logit_w_score, int_logit_avg_score, result_df = get_score(
        model=model,
        tokenized_inputs=tokenized_inputs,
        tokenizer=tokenizer,
        weights=weights,
        temperature=0,
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        score_token="level",
    )

    print(direct_score, weighted_score, int_logit_w_score, int_logit_avg_score)

    layer_results = None if result_df is None else result_df.to_dict()

    return {
        "direct": direct_score,
        "weighted": weighted_score,
        "int_logit_w_score": int_logit_w_score,
        "int_logit_avg_score": int_logit_avg_score,
        "layers": str(layer_results),
    }