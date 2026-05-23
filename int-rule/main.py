from tqdm import tqdm
import os
import random
import numpy as np
import torch

from optimize_layer_weights import optimize_layer_weights
from utils_compare import (
    load_H_targets,
    load_baseline_scores,
    ordinal_probs,
    fit_eta_z_from_P,
    train_bridge_alternating,
    train_bridge_from_probs,
    train_bridge_from_last_probs_mean,
    pred_human,
    pred_human_from_probs,
    pred_human_from_last_probs_mean,
    print_correlations,
    print_prob_metrics,
    print_mean_metrics,
    print_uncertainty_metrics,
)


def set_global_seed(seed):
    """Set random seeds for one independent optimization run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # =========================
    # 1. Paths and settings
    # =========================

    result_root = os.environ.get("RESULT_ROOT", "/path/to/results")
    valid_root = os.path.join(result_root, "valid")

    compare_save_dir = os.path.join(result_root, "compare_rerun")
    os.makedirs(compare_save_dir, exist_ok=True)

    analysis_save_dir = os.path.join(compare_save_dir, "analysis_payload")
    os.makedirs(analysis_save_dir, exist_ok=True)

    api_compare_save_dir = os.path.join(result_root, "compare_api_test")
    os.makedirs(api_compare_save_dir, exist_ok=True)

    api_analysis_save_dir = os.path.join(api_compare_save_dir, "analysis_payload")
    os.makedirs(api_analysis_save_dir, exist_ok=True)

    dataset_names = [
        "biggen",
        "chatbot_arena",
        "flask",
        "helpsteer",
        "mt_bench",
    ]

    methods = ["random"]
    reset_csv = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 2. Hyperparameters
    # =========================

    # Calibration sizes used in the paper.
    sizes = [20, 40, 80, 160, 320]

    # Five independent optimization seeds.
    seeds = [1, 2, 3, 4, 5]

    # Layer-weight initialization.
    layer_weight_epochs = 2
    layer_weight_lr = 1e-2
    layer_weight_min_lr = 1e-3
    layer_weight_batch_size = 8

    # Model-side latent initialization.
    init_num_steps = 2000
    init_lr = 1e-2
    init_use_lbfgs = True

    # Rule recalibration.
    num_outer_iters = 50
    num_steps_A = 100
    num_steps_B = 100
    num_steps_C = 100
    lr_A = 1e-2
    lr_B = 1e-2
    lr_C = 1e-2

    # =========================
    # 3. Metric output paths
    # =========================

    correlation_csv_path = os.path.join(compare_save_dir, "correlation.csv")
    prob_csv_path = os.path.join(compare_save_dir, "prob_metrics.csv")
    mean_csv_path = os.path.join(compare_save_dir, "mean_metrics.csv")
    uncertainty_csv_path = os.path.join(compare_save_dir, "uncertainty_metrics.csv")

    api_correlation_csv_path = os.path.join(api_compare_save_dir, "correlation.csv")
    api_prob_csv_path = os.path.join(api_compare_save_dir, "prob_metrics.csv")
    api_mean_csv_path = os.path.join(api_compare_save_dir, "mean_metrics.csv")
    api_uncertainty_csv_path = os.path.join(api_compare_save_dir, "uncertainty_metrics.csv")

    if reset_csv:
        for path in [
            correlation_csv_path,
            prob_csv_path,
            mean_csv_path,
            uncertainty_csv_path,
            api_correlation_csv_path,
            api_prob_csv_path,
            api_mean_csv_path,
            api_uncertainty_csv_path,
        ]:
            if os.path.exists(path):
                os.remove(path)

    # =========================
    # 4. Build all evaluation tasks
    # =========================

    all_tasks = []

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(result_root, dataset_name)

        if not os.path.isdir(dataset_dir):
            continue

        model_files = [
            fname
            for fname in os.listdir(dataset_dir)
            if (
                fname.endswith("_logits.json")
                or fname.endswith("_with_feedback_logits.json")
                or fname.endswith("_api_logprobs.json")
                or fname.endswith("_with_feedback_api_logprobs.json")
            )
        ]

        # Optional model subset. Set to None to run all available model files.
        target_model_files = None

        # Example:
        # target_model_files = {
        #     "Model-Name_logits.json",
        #     "Model-Name_with_feedback_logits.json",
        # }

        if target_model_files is not None:
            model_files = [fname for fname in model_files if fname in target_model_files]

        for fname in model_files:
            data_path = os.path.join(dataset_dir, fname)
            prefix = os.path.splitext(os.path.basename(data_path))[0]

            valid_data_path_dict = {}

            for method in methods:
                if method == "raw":
                    path = os.path.join(valid_root, f"{prefix}_{dataset_name}.json")
                    if os.path.exists(path):
                        valid_data_path_dict["raw"] = path
                else:
                    for size in sizes:
                        path = os.path.join(
                            valid_root,
                            dataset_name,
                            f"{prefix}_{method}_{size}.json",
                        )
                        if os.path.exists(path):
                            valid_data_path_dict[f"{method}_{size}"] = path

            for valid_name, valid_data_path in valid_data_path_dict.items():
                for seed in seeds:
                    all_tasks.append(
                        (
                            dataset_name,
                            fname,
                            data_path,
                            prefix,
                            valid_name,
                            valid_data_path,
                            seed,
                        )
                    )

    # =========================
    # 5. Run all tasks
    # =========================

    print(f"Total tasks = {len(all_tasks)}")
    global_pbar = tqdm(all_tasks, desc="Total Progress", ncols=140)

    eval_cache = {}

    for (
        dataset_name,
        fname,
        data_path,
        prefix,
        valid_name,
        valid_data_path,
        seed,
    ) in global_pbar:
        valid_name_with_seed = f"{valid_name}_seed{seed}"
        set_global_seed(seed)

        global_pbar.set_postfix(
            dataset=dataset_name,
            model=prefix[:28],
            valid=valid_name,
            seed=seed,
        )

        print("\n" + "=" * 100)
        print(f"dataset = {dataset_name}")
        print(f"model_file = {fname}")
        print(f"data_path = {data_path}")
        print(f"valid_name = {valid_name}")
        print(f"seed = {seed}")
        print(f"valid_data_path = {valid_data_path}")
        print("=" * 100)

        try:
            # =========================
            # 6. Label-space settings
            # =========================

            label_start = 1
            data_path_lower = data_path.lower()

            if (
                "mt_bench" in data_path_lower
                or "mt-bench" in data_path_lower
                or "arena" in data_path_lower
            ):
                label_start = 0

            is_api_model = (
                data_path_lower.endswith("_api_logprobs.json")
                or data_path_lower.endswith("_with_feedback_api_logprobs.json")
            )

            drop_last_layer = not is_api_model

            cur_analysis_save_dir = api_analysis_save_dir if is_api_model else analysis_save_dir
            cur_correlation_csv_path = (
                api_correlation_csv_path if is_api_model else correlation_csv_path
            )
            cur_prob_csv_path = api_prob_csv_path if is_api_model else prob_csv_path
            cur_mean_csv_path = api_mean_csv_path if is_api_model else mean_csv_path
            cur_uncertainty_csv_path = (
                api_uncertainty_csv_path if is_api_model else uncertainty_csv_path
            )

            analysis_save_path = os.path.join(
                cur_analysis_save_dir,
                f"{prefix}_{dataset_name}_{valid_name_with_seed}_analysis.pt",
            )

            if os.path.exists(analysis_save_path):
                print(f"[skip] analysis payload already exists: {analysis_save_path}")
                continue

            # =========================
            # 7. Load evaluation data
            # =========================

            cache_key = (dataset_name, fname)

            if cache_key not in eval_cache:
                (
                    H_eval,
                    target_probs_eval,
                    target_mean_eval,
                    target_label_eval,
                    target_median_eval,
                ) = load_H_targets(
                    data_path,
                    label_start,
                    drop_last_layer=drop_last_layer,
                    device=device,
                )

                raw_label, raw_score, raw_probs = load_baseline_scores(
                    data_path,
                    label_start,
                    device=device,
                )

                eval_cache[cache_key] = {
                    "H_eval": H_eval,
                    "target_probs_eval": target_probs_eval,
                    "target_mean_eval": target_mean_eval,
                    "target_label_eval": target_label_eval,
                    "target_median_eval": target_median_eval,
                    "raw_label": raw_label,
                    "raw_score": raw_score,
                    "raw_probs": raw_probs,
                }

            H_eval = eval_cache[cache_key]["H_eval"]
            target_probs_eval = eval_cache[cache_key]["target_probs_eval"]
            target_mean_eval = eval_cache[cache_key]["target_mean_eval"]
            target_label_eval = eval_cache[cache_key]["target_label_eval"]
            target_median_eval = eval_cache[cache_key]["target_median_eval"]
            raw_label = eval_cache[cache_key]["raw_label"]
            raw_score = eval_cache[cache_key]["raw_score"]
            raw_probs = eval_cache[cache_key]["raw_probs"]

            # =========================
            # 8. Learn or assign layer weights
            # =========================

            if is_api_model:
                layer_weights = torch.ones(
                    H_eval.size(1),
                    device=device,
                    dtype=H_eval.dtype,
                )
            else:
                layer_weights = optimize_layer_weights(
                    valid_data_path,
                    label_start=label_start,
                    num_epochs=layer_weight_epochs,
                    lr=layer_weight_lr,
                    min_lr=layer_weight_min_lr,
                    batch_size=layer_weight_batch_size,
                    seed=seed,
                ).to(device)

            print("is_api_model =", is_api_model)
            print("drop_last_layer =", drop_last_layer)
            print("H_eval.shape =", H_eval.shape)
            print("layer_weights =", layer_weights)
            print("layer_weights.sum() =", layer_weights.sum().item())
            print("=" * 60)

            # =========================
            # 9. Load calibration data
            # =========================

            (
                H_valid,
                target_probs_valid,
                target_mean_valid,
                target_label_valid,
                target_median_valid,
            ) = load_H_targets(
                valid_data_path,
                label_start,
                drop_last_layer=drop_last_layer,
                device=device,
            )

            print("H_valid.shape =", H_valid.shape)
            print("layer_weights.shape =", layer_weights.shape)

            if H_valid.size(1) != layer_weights.numel():
                raise ValueError(
                    f"Layer mismatch: H_valid L={H_valid.size(1)}, "
                    f"layer_weights={layer_weights.numel()}"
                )

            if H_eval.size(1) != layer_weights.numel():
                raise ValueError(
                    f"Layer mismatch: H_eval L={H_eval.size(1)}, "
                    f"layer_weights={layer_weights.numel()}"
                )

            _, _, raw_probs_valid = load_baseline_scores(
                valid_data_path,
                label_start,
                device=device,
            )

            print("raw_probs_valid.shape =", raw_probs_valid.shape)
            print("target_probs_valid.shape =", target_probs_valid.shape)

            if raw_probs_valid.size(0) != target_probs_valid.size(0):
                raise ValueError(
                    f"Sample-size mismatch: raw_probs_valid={raw_probs_valid.size(0)}, "
                    f"target_probs_valid={target_probs_valid.size(0)}"
                )

            if raw_probs_valid.size(1) != target_probs_valid.size(1):
                raise ValueError(
                    f"Class-size mismatch: raw_probs_valid={raw_probs_valid.size(1)}, "
                    f"target_probs_valid={target_probs_valid.size(1)}"
                )

            # =========================
            # 10. Initialize latent variables for Int-Rule++
            # =========================

            if not is_api_model:
                int_logit_w_logits_valid = (
                    H_valid * layer_weights.view(1, -1, 1)
                ).sum(dim=1)
                int_logit_w_probs_valid = torch.softmax(
                    int_logit_w_logits_valid,
                    dim=1,
                )

                init_result = fit_eta_z_from_P(
                    int_logit_w_probs_valid,
                    num_steps=init_num_steps,
                    lr=init_lr,
                    use_lbfgs=init_use_lbfgs,
                    seed=seed,
                    verbose=True,
                )

                eta_init = init_result["eta_init"]
                z_init = init_result["z_init"]

                # =========================
                # 11. Train Int-Rule++
                # =========================

                int_rule_pp_result = train_bridge_alternating(
                    H_valid,
                    target_probs_valid,
                    num_outer_iters=num_outer_iters,
                    num_steps_A=num_steps_A,
                    num_steps_B=num_steps_B,
                    num_steps_C=num_steps_C,
                    lr_A=lr_A,
                    lr_B=lr_B,
                    lr_C=lr_C,
                    lambda_judge=1.0,
                    lambda_human=1.0,
                    lambda_ord=0.0,
                    ord_type="emd",
                    w_init=layer_weights,
                    eta_init=eta_init,
                    z_init=z_init,
                )

                print("\nInt-Rule++ training finished")
                print("final layer weights =", int_rule_pp_result["w"])
                print("final model-side thresholds =", int_rule_pp_result["eta"])
                print("final human-side thresholds =", int_rule_pp_result["alpha"])
                print("final scale =", int_rule_pp_result["beta"])
                print("=" * 60)

            else:
                int_logit_w_probs_valid = None
                int_rule_pp_result = None

            # =========================
            # 12. Train Final-Rule++
            # =========================

            final_rule_pp_init = fit_eta_z_from_P(
                raw_probs_valid,
                num_steps=init_num_steps,
                lr=init_lr,
                use_lbfgs=init_use_lbfgs,
                seed=seed,
                verbose=True,
            )

            final_rule_pp_result = train_bridge_from_probs(
                P=raw_probs_valid,
                target_probs=target_probs_valid,
                num_outer_iters=num_outer_iters,
                num_steps_A=num_steps_A,
                num_steps_B=num_steps_B,
                lr_A=lr_A,
                lr_B=lr_B,
                lambda_ord=0.0,
                ord_type="emd",
                eta_init=final_rule_pp_init["eta_init"],
                z_init=final_rule_pp_init["z_init"],
                verbose=True,
            )

            print("\nFinal-Rule++ training finished")
            print("Final-Rule++ model-side thresholds =", final_rule_pp_result["eta"])
            print("Final-Rule++ human-side thresholds =", final_rule_pp_result["alpha"])
            print("Final-Rule++ scale =", final_rule_pp_result["beta"])
            print("=" * 60)

            with torch.no_grad():
                eta = final_rule_pp_result["eta"].to(device)
                z_l = final_rule_pp_result["z_l"].to(device)
                final_rule_pp_recon_valid = ordinal_probs(eta, z_l)

                print(
                    "Final-Rule++ valid reconstruction MAE =",
                    torch.abs(final_rule_pp_recon_valid - raw_probs_valid).mean().item(),
                )
                print("raw_probs_valid mean =", raw_probs_valid.mean(dim=0))
                print(
                    "Final-Rule++ reconstruction mean =",
                    final_rule_pp_recon_valid.mean(dim=0),
                )

            # =========================
            # 13. Train Final-Rule
            # =========================

            final_rule_result = train_bridge_from_last_probs_mean(
                P=raw_probs_valid,
                target_probs=target_probs_valid,
                num_outer_iters=num_outer_iters,
                num_steps_B=num_steps_B,
                lr_B=lr_B,
                lambda_ord=0.0,
                ord_type="emd",
                verbose=True,
            )

            print("\nFinal-Rule training finished")
            print("Final-Rule human-side thresholds =", final_rule_result["alpha"])
            print("Final-Rule scale =", final_rule_result["beta"])
            print("=" * 60)

            # =========================
            # 14. Train Int-Rule
            # =========================

            if not is_api_model:
                int_rule_result = train_bridge_from_last_probs_mean(
                    P=int_logit_w_probs_valid,
                    target_probs=target_probs_valid,
                    num_outer_iters=num_outer_iters,
                    num_steps_B=num_steps_B,
                    lr_B=lr_B,
                    lambda_ord=0.0,
                    ord_type="emd",
                    verbose=True,
                )

                print("\nInt-Rule training finished")
                print("Int-Rule human-side thresholds =", int_rule_result["alpha"])
                print("Int-Rule scale =", int_rule_result["beta"])
                print("=" * 60)

            else:
                int_rule_result = None

            # =========================
            # 15. Predict on the evaluation set
            # =========================

            (
                z_final_rule_pp,
                final_rule_pp_probs,
                final_rule_pp_label,
                final_rule_pp_score,
            ) = pred_human_from_probs(
                raw_probs,
                final_rule_pp_result,
                z_init=None,
                verbose=True,
            )

            (
                z_final_rule,
                final_rule_probs,
                final_rule_label,
                final_rule_score,
            ) = pred_human_from_last_probs_mean(
                raw_probs,
                final_rule_result,
                verbose=True,
            )

            print(
                "Raw argmax counts =",
                torch.bincount(
                    raw_probs.argmax(dim=1).long(),
                    minlength=raw_probs.size(1),
                ),
            )
            print(
                "Final-Rule++ label counts =",
                torch.bincount(
                    final_rule_pp_label.long(),
                    minlength=final_rule_pp_probs.size(1),
                ),
            )
            print(
                "Target label counts =",
                torch.bincount(
                    target_label_eval.long(),
                    minlength=final_rule_pp_probs.size(1),
                ),
            )
            print(
                "Final-Rule++ score mean/std =",
                final_rule_pp_score.mean().item(),
                final_rule_pp_score.std().item(),
            )
            print("Final-Rule++ probability mean =", final_rule_pp_probs.mean(dim=0))

            # =========================
            # 16. Build method outputs
            # =========================

            if is_api_model:
                all_score_dict = {
                    "final_rule_pp_label": final_rule_pp_label,
                    "final_rule_pp_score": final_rule_pp_score,
                    "final_rule_label": final_rule_label,
                    "final_rule_score": final_rule_score,
                    "raw_label": raw_label,
                    "raw_score": raw_score,
                }

                prob_dict = {
                    "Final-Rule++": final_rule_pp_probs,
                    "Final-Rule": final_rule_probs,
                    "Raw": raw_probs,
                }

                score_dict = {
                    "Final-Rule++": final_rule_pp_score,
                    "Final-Rule": final_rule_score,
                    "Raw": raw_score,
                }

            else:
                (
                    z_int_rule_pp,
                    int_rule_pp_probs,
                    int_rule_pp_label,
                    int_rule_pp_score,
                ) = pred_human(
                    H_eval,
                    int_rule_pp_result,
                    z_init=None,
                    verbose=True,
                )

                with torch.no_grad():
                    int_logit_w_logits_eval = (
                        H_eval * layer_weights.view(1, -1, 1)
                    ).sum(dim=1)
                    int_logit_w_probs_eval = torch.softmax(
                        int_logit_w_logits_eval,
                        dim=1,
                    )

                (
                    z_int_rule,
                    int_rule_probs,
                    int_rule_label,
                    int_rule_score,
                ) = pred_human_from_last_probs_mean(
                    int_logit_w_probs_eval,
                    int_rule_result,
                    verbose=True,
                )

                score_values = torch.arange(
                    H_eval.size(2),
                    device=device,
                    dtype=H_eval.dtype,
                ).view(1, -1)

                # Int-Logit-W: weighted aggregation over layer-wise logits.
                int_logit_w_logits = (
                    H_eval * layer_weights.view(1, -1, 1)
                ).sum(dim=1)
                int_logit_w_probs = torch.softmax(int_logit_w_logits, dim=1)
                int_logit_w_label = torch.argmax(int_logit_w_probs, dim=1)
                int_logit_w_score = (int_logit_w_probs * score_values).sum(dim=1)

                # Int-Prob-W: weighted aggregation over layer-wise probabilities.
                int_prob_w_probs = (
                    torch.softmax(H_eval, dim=2) * layer_weights.view(1, -1, 1)
                ).sum(dim=1)
                int_prob_w_label = torch.argmax(int_prob_w_probs, dim=1)
                int_prob_w_score = (int_prob_w_probs * score_values).sum(dim=1)

                # Int-Logit-Avg: uniform aggregation over layer-wise logits.
                int_logit_avg_logits = H_eval.mean(dim=1)
                int_logit_avg_probs = torch.softmax(int_logit_avg_logits, dim=1)
                int_logit_avg_label = torch.argmax(int_logit_avg_probs, dim=1)
                int_logit_avg_score = (int_logit_avg_probs * score_values).sum(dim=1)

                # Int-Prob-Avg: uniform aggregation over layer-wise probabilities.
                int_prob_avg_probs = torch.softmax(H_eval, dim=2).mean(dim=1)
                int_prob_avg_label = torch.argmax(int_prob_avg_probs, dim=1)
                int_prob_avg_score = (int_prob_avg_probs * score_values).sum(dim=1)

                all_score_dict = {
                    "int_rule_pp_label": int_rule_pp_label,
                    "int_rule_pp_score": int_rule_pp_score,
                    "final_rule_pp_label": final_rule_pp_label,
                    "final_rule_pp_score": final_rule_pp_score,
                    "final_rule_label": final_rule_label,
                    "final_rule_score": final_rule_score,
                    "int_rule_label": int_rule_label,
                    "int_rule_score": int_rule_score,
                    "int_logit_w_label": int_logit_w_label,
                    "int_logit_w_score": int_logit_w_score,
                    "int_prob_w_label": int_prob_w_label,
                    "int_prob_w_score": int_prob_w_score,
                    "int_logit_avg_label": int_logit_avg_label,
                    "int_logit_avg_score": int_logit_avg_score,
                    "int_prob_avg_label": int_prob_avg_label,
                    "int_prob_avg_score": int_prob_avg_score,
                    "raw_label": raw_label,
                    "raw_score": raw_score,
                }

                prob_dict = {
                    "Int-Rule++": int_rule_pp_probs,
                    "Final-Rule++": final_rule_pp_probs,
                    "Final-Rule": final_rule_probs,
                    "Int-Rule": int_rule_probs,
                    "Int-Logit-W": int_logit_w_probs,
                    "Int-Prob-W": int_prob_w_probs,
                    "Int-Logit-Avg": int_logit_avg_probs,
                    "Int-Prob-Avg": int_prob_avg_probs,
                    "Raw": raw_probs,
                }

                score_dict = {
                    "Int-Rule++": int_rule_pp_score,
                    "Final-Rule++": final_rule_pp_score,
                    "Final-Rule": final_rule_score,
                    "Int-Rule": int_rule_score,
                    "Int-Logit-W": int_logit_w_score,
                    "Int-Prob-W": int_prob_w_score,
                    "Int-Logit-Avg": int_logit_avg_score,
                    "Int-Prob-Avg": int_prob_avg_score,
                    "Raw": raw_score,
                }

            # =========================
            # 17. Check output-target alignment
            # =========================

            for method_name, value in prob_dict.items():
                if isinstance(value, torch.Tensor) and value.size(0) != target_probs_eval.size(0):
                    raise ValueError(
                        f"{method_name} sample-size mismatch: "
                        f"pred={value.size(0)}, target={target_probs_eval.size(0)}"
                    )

            for method_name, value in score_dict.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.view(-1).size(0) != target_mean_eval.view(-1).size(0)
                ):
                    raise ValueError(
                        f"{method_name} sample-size mismatch: "
                        f"pred={value.view(-1).size(0)}, "
                        f"target={target_mean_eval.view(-1).size(0)}"
                    )

            # =========================
            # 18. Save analysis payload
            # =========================

            analysis_payload = {
                "meta": {
                    "model_name": prefix,
                    "dataset_name": dataset_name,
                    "valid_name": valid_name_with_seed,
                    "base_valid_name": valid_name,
                    "seed": seed,
                    "data_file": os.path.basename(data_path),
                    "valid_file": os.path.basename(valid_data_path),
                    "label_start": label_start,
                    "is_api_model": is_api_model,
                    "drop_last_layer": drop_last_layer,
                    "hyperparameters": {
                        "layer_weight_epochs": layer_weight_epochs,
                        "layer_weight_lr": layer_weight_lr,
                        "layer_weight_min_lr": layer_weight_min_lr,
                        "layer_weight_batch_size": layer_weight_batch_size,
                        "init_num_steps": init_num_steps,
                        "init_lr": init_lr,
                        "init_use_lbfgs": init_use_lbfgs,
                        "num_outer_iters": num_outer_iters,
                        "num_steps_A": num_steps_A,
                        "num_steps_B": num_steps_B,
                        "num_steps_C": num_steps_C,
                        "lr_A": lr_A,
                        "lr_B": lr_B,
                        "lr_C": lr_C,
                    },
                },
                "targets": {
                    "target_probs": target_probs_eval.detach().cpu(),
                    "target_mean": target_mean_eval.detach().cpu(),
                    "target_label": target_label_eval.detach().cpu(),
                    "target_median": target_median_eval.detach().cpu(),
                },
                "probs": {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in prob_dict.items()
                },
                "scores": {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in score_dict.items()
                },
            }

            torch.save(analysis_payload, analysis_save_path)
            print("analysis saved to:", analysis_save_path)

            # =========================
            # 19. Save metric tables
            # =========================

            print_correlations(
                all_score_dict,
                target_mean_eval,
                save_csv_path=cur_correlation_csv_path,
                model_name=prefix,
                dataset_name=dataset_name,
                valid_name=valid_name_with_seed,
            )

            print_prob_metrics(
                prob_dict,
                target_label=target_label_eval,
                target_probs=target_probs_eval,
                target_median=target_median_eval,
                n_bins=10,
                strategy="quantile",
                save_csv_path=cur_prob_csv_path,
                model_name=prefix,
                dataset_name=dataset_name,
                valid_name=valid_name_with_seed,
            )

            print_mean_metrics(
                score_dict,
                target_mean=target_mean_eval,
                n_bins=10,
                strategy="quantile",
                save_csv_path=cur_mean_csv_path,
                model_name=prefix,
                dataset_name=dataset_name,
                valid_name=valid_name_with_seed,
            )

            # =========================
            # 20. Save uncertainty metrics
            # =========================

            if dataset_name in ["flask", "mt_bench", "mt-bench", "mtbench"]:
                print_uncertainty_metrics(
                    all_score_dict=all_score_dict,
                    prob_dict=prob_dict,
                    score_dict=score_dict,
                    target_mean=target_mean_eval,
                    target_label=target_label_eval,
                    target_probs=target_probs_eval,
                    target_median=target_median_eval,
                    model_name=prefix,
                    dataset_name=dataset_name,
                    valid_name=valid_name_with_seed,
                    scheme_name="main",
                    n_bins=10,
                    strategy="quantile",
                    save_csv_path=cur_uncertainty_csv_path,
                )

                print_uncertainty_metrics(
                    all_score_dict=all_score_dict,
                    prob_dict=prob_dict,
                    score_dict=score_dict,
                    target_mean=target_mean_eval,
                    target_label=target_label_eval,
                    target_probs=target_probs_eval,
                    target_median=target_median_eval,
                    model_name=prefix,
                    dataset_name=dataset_name,
                    valid_name=valid_name_with_seed,
                    scheme_name="appendix",
                    n_bins=10,
                    strategy="quantile",
                    save_csv_path=cur_uncertainty_csv_path,
                )

        except Exception as error:
            print(
                f"[ERROR] dataset={dataset_name}, "
                f"model_file={fname}, "
                f"valid_name={valid_name}, "
                f"seed={seed}"
            )
            print(error)
            continue

    global_pbar.close()

    print("\nAll tasks finished")
    print("correlation csv:", correlation_csv_path)
    print("prob csv:", prob_csv_path)
    print("mean csv:", mean_csv_path)
    print("analysis dir:", analysis_save_dir)
    print("uncertainty csv:", uncertainty_csv_path)
    print("api correlation csv:", api_correlation_csv_path)
    print("api prob csv:", api_prob_csv_path)
    print("api mean csv:", api_mean_csv_path)
    print("api analysis dir:", api_analysis_save_dir)
    print("api uncertainty csv:", api_uncertainty_csv_path)