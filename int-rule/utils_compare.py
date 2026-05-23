import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter
from torch.optim import Adam
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve


def load_H_y(data_path, zero_based=True, drop_last_layer=True, device=None):
    """
    Load layer-wise logits and human labels from a JSON result file.

    Args:
        data_path: Path to the JSON result file.
        zero_based: Whether to convert labels from 1-based to 0-based.
        drop_last_layer: Whether to remove the final layer logits.
        device: Optional torch device.

    Returns:
        H: Tensor with shape [N, L, K].
        y_h: Tensor with shape [N].
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    logits_list = []
    human_score_list = []

    for sample in all_data:
        if "weighted_socre" in sample and sample["weighted_socre"] == -1:
            continue

        df = pd.DataFrame(sample["df"])
        logits = torch.tensor([x for x in df["logits"]], dtype=torch.float32)
        human_score = torch.tensor(sample["human_score"], dtype=torch.long)

        if drop_last_layer:
            logits = logits[:-1, :]

        logits_list.append(logits)
        human_score_list.append(human_score)

    H = torch.stack(logits_list, dim=0)
    y_h = torch.stack(human_score_list, dim=0)

    if zero_based:
        y_h = y_h - 1

    if device is not None:
        H = H.to(device)
        y_h = y_h.to(device)

    return H, y_h


def load_H_targets(data_path, label_start, drop_last_layer=True, device=None):
    """
    Load layer-wise logits and convert human annotations into distributions.

    Args:
        data_path: Path to the JSON result file.
        label_start: Label offset. Use 0 for 0-based labels and 1 for 1-based labels.
        drop_last_layer: Whether to remove the final layer logits.
        device: Optional torch device.

    Returns:
        H: Tensor with shape [N, L, K].
        target_probs: Human rating distributions with shape [N, K].
        target_mean: Expected human scores with shape [N].
        target_label: Mode labels with shape [N].
        target_median: Median labels with shape [N].
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    logits_list = []
    probs_list = []
    mean_list = []
    label_list = []
    median_list = []

    for sample in all_data:
        if "weighted_socre" in sample and sample["weighted_socre"] == -1:
            continue

        if "human_score" in sample:
            if sample["human_score"] == -1:
                continue
            if isinstance(sample["human_score"], list) and any(s == -1 for s in sample["human_score"]):
                continue

        df = pd.DataFrame(sample["df"])
        logits = torch.tensor([x for x in df["logits"]], dtype=torch.float32)

        if drop_last_layer:
            logits = logits[:-1, :]

        num_classes = logits.size(1)

        scores = sample["human_score"]
        if not isinstance(scores, list):
            scores = [scores]

        scores = torch.tensor(scores, dtype=torch.long) - label_start

        if scores.dim() != 1:
            raise ValueError(f"scores must be 1-D, got shape={scores.shape}")

        if torch.any(scores < 0):
            raise ValueError(
                f"negative labels found: raw={sample['human_score']}, "
                f"label_start={label_start}, shifted={scores.tolist()}"
            )

        counts = torch.bincount(scores, minlength=num_classes).float()
        probs = counts / counts.sum()

        levels = torch.arange(num_classes, dtype=torch.float32)
        mean = (probs * levels).sum()

        label = counts.argmax()
        median = scores.median()

        logits_list.append(logits)
        probs_list.append(probs)
        mean_list.append(mean)
        label_list.append(label)
        median_list.append(median)

    H = torch.stack(logits_list)
    target_probs = torch.stack(probs_list)
    target_mean = torch.stack(mean_list)
    target_label = torch.stack(label_list)
    target_median = torch.stack(median_list)

    if device is not None:
        H = H.to(device)
        target_probs = target_probs.to(device)
        target_mean = target_mean.to(device)
        target_label = target_label.to(device)
        target_median = target_median.to(device)

    return H, target_probs, target_mean, target_label, target_median


def load_baseline_scores(data_path, label_start, device=None):
    """
    Load Raw scores and final-layer probability distributions.

    The original result files use legacy keys such as `direct_socre` and
    `weighted_socre`, so these keys are kept for compatibility.

    Returns:
        direct_score: Raw argmax scores with shape [N].
        e_score: Raw expected scores with shape [N].
        last_probs: Final-layer probability distributions with shape [N, K].
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    direct_score_list = []
    e_score_list = []
    last_probs_list = []

    for sample in all_data:
        if sample["weighted_socre"] == -1:
            continue

        if "human_score" in sample:
            if sample["human_score"] == -1:
                continue
            if isinstance(sample["human_score"], list) and any(s == -1 for s in sample["human_score"]):
                continue

        direct_score_list.append(sample["direct_socre"] - label_start)
        e_score_list.append(sample["weighted_socre"] - label_start)

        probs_dict = sample["df"]["probs"]
        last_idx = str(max(int(k) for k in probs_dict.keys()))
        last_probs_list.append(probs_dict[last_idx])

    if device is not None:
        direct_score = torch.tensor(direct_score_list, dtype=torch.float32, device=device)
        e_score = torch.tensor(e_score_list, dtype=torch.float32, device=device)
        last_probs = torch.tensor(last_probs_list, dtype=torch.float32, device=device)
    else:
        direct_score = np.array(direct_score_list, dtype=float)
        e_score = np.array(e_score_list, dtype=float)
        last_probs = np.array(last_probs_list, dtype=float)

    return direct_score, e_score, last_probs


def build_eta(free_cutoffs):
    """
    Build ordered model-side thresholds.

    The first threshold is fixed to 0 for identifiability, and the remaining
    thresholds are generated by positive gaps.
    """
    first = torch.zeros(1, dtype=free_cutoffs.dtype, device=free_cutoffs.device)
    gaps = F.softplus(free_cutoffs) + 1e-6
    rest = torch.cumsum(gaps, dim=0)
    return torch.cat([first, rest], dim=0)


def build_alpha(alpha_free):
    """
    Build ordered human-side thresholds.

    The first threshold is free, and the remaining thresholds are generated
    by positive gaps.
    """
    first = alpha_free[:1]
    gaps = F.softplus(alpha_free[1:]) + 1e-6
    rest = first + torch.cumsum(gaps, dim=0)
    return torch.cat([first, rest], dim=0)


def ordinal_probs(cutoffs, z):
    """
    Compute ordered-threshold probabilities.

    Args:
        cutoffs: Ordered thresholds with shape [K-1].
        z: Latent scores with shape [N] or [N, 1].

    Returns:
        probs: Label probabilities with shape [N, K].
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)

    cutoffs = cutoffs.view(1, -1)
    cdf = torch.sigmoid(cutoffs - z)

    left = torch.zeros((z.size(0), 1), dtype=z.dtype, device=z.device)
    right = torch.ones((z.size(0), 1), dtype=z.dtype, device=z.device)

    cdf_full = torch.cat([left, cdf, right], dim=1)
    probs = torch.diff(cdf_full, dim=1)

    probs = probs.clamp_min(1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def pred_human(
    H,
    result,
    z_init=None,
    beta_smoothl1=1e-3,
    num_epochs=2000,
    lr=1e-2,
    tol_loss=1e-10,
    tol_grad=1e-10,
    scheduler_factor=0.5,
    patience=50,
    lbfgs_max_iter=200,
    lbfgs_lr=1.0,
    verbose=False,
):
    """
    Predict human-side distributions for Int-Rule++.

    The function first reconstructs the model-side latent score from the
    aggregated model-side distribution, then applies the learned human-side
    ordered-threshold rule.
    """
    device = H.device

    w = result["w"].to(device).double()
    eta = result["eta"].to(device).double()
    alpha = result["alpha"].to(device).double()
    beta = result["beta"].to(device).double()

    fused_logits = (H.double() * w.view(1, -1, 1)).sum(dim=1)
    model_probs = torch.softmax(fused_logits, dim=1)

    num_samples = model_probs.shape[0]

    if z_init is None:
        z_init = torch.zeros(num_samples, device=device, dtype=torch.float64)
    else:
        z_init = z_init.detach().to(device).double()

    z_l = nn.Parameter(z_init[:, None])

    optimizer = optim.Adam([z_l], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=patience,
    )

    for _ in range(num_epochs):
        optimizer.zero_grad()

        pred_model_probs = ordinal_probs(eta.float(), z_l.squeeze(1).float()).double()
        loss = F.smooth_l1_loss(pred_model_probs, model_probs, beta=beta_smoothl1)

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        final_loss = torch.abs(pred_model_probs - model_probs).mean()
        grad_norm = z_l.grad.norm().item()

        if final_loss.item() < tol_loss or grad_norm < tol_grad:
            break

    lbfgs_optimizer = optim.LBFGS(
        [z_l],
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        pred_model_probs = ordinal_probs(eta.float(), z_l.squeeze(1).float()).double()
        loss = F.smooth_l1_loss(pred_model_probs, model_probs, beta=beta_smoothl1)
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)

    with torch.no_grad():
        z_l_pred = z_l.squeeze(1)
        z_h = z_l_pred / beta

        pred_probs = ordinal_probs(alpha.float(), z_h.float()).double()
        pred_label = pred_probs.argmax(dim=1)

        levels = torch.arange(
            pred_probs.size(1),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )
        pred_score = (pred_probs * levels.view(1, -1)).sum(dim=1)

        if verbose:
            pred_model_probs = ordinal_probs(eta.float(), z_l_pred.float()).double()
            recon_mae = torch.abs(pred_model_probs - model_probs).mean().item()
            print("Int-Rule++ model-side reconstruction MAE:", recon_mae)

    return z_l_pred.float(), pred_probs.float(), pred_label, pred_score.float()


def fit_eta_z_from_P(
    P,
    num_steps=2000,
    lr=1e-2,
    beta_smoothl1=1e-3,
    tol_loss=1e-8,
    tol_grad=1e-8,
    scheduler_factor=0.5,
    patience=50,
    use_lbfgs=True,
    lbfgs_max_iter=200,
    lbfgs_lr=1.0,
    seed=42,
    verbose=False,
):
    """
    Fit model-side thresholds and latent scores to reconstruct a distribution P.

    This initialization is trained for 2000 steps with learning rate 1e-2,
    followed by L-BFGS refinement when enabled.
    """
    torch.manual_seed(seed)

    device = P.device
    P = P.double()
    num_samples, num_classes = P.shape

    eta_free = nn.Parameter(torch.zeros(num_classes - 2, device=device, dtype=torch.float64))
    z_l = nn.Parameter(torch.zeros(num_samples, device=device, dtype=torch.float64))

    optimizer = optim.Adam([eta_free, z_l], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=patience,
    )

    for _ in range(num_steps):
        optimizer.zero_grad()

        eta = build_eta(eta_free).double()
        pred = ordinal_probs(eta.float(), z_l.float()).double()

        loss = F.smooth_l1_loss(pred, P, beta=beta_smoothl1)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        fit_mae = torch.abs(pred - P).mean()
        grad_norm = (eta_free.grad.norm() ** 2 + z_l.grad.norm() ** 2).sqrt().item()

        if fit_mae.item() < tol_loss or grad_norm < tol_grad:
            break

    if use_lbfgs:
        lbfgs = optim.LBFGS(
            [eta_free, z_l],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            eta = build_eta(eta_free).double()
            pred = ordinal_probs(eta.float(), z_l.float()).double()
            loss = F.smooth_l1_loss(pred, P, beta=beta_smoothl1)
            loss.backward()
            return loss

        lbfgs.step(closure)

    with torch.no_grad():
        eta = build_eta(eta_free).float()
        pred = ordinal_probs(eta, z_l.float())
        fit_mae = torch.abs(pred.double() - P).mean().item()

    if verbose:
        print(f"initial eta/z fitting MAE = {fit_mae:.8f}")

    return {
        "eta_init": eta.detach(),
        "z_init": z_l.detach().float(),
        "fit_probs": pred.detach(),
        "fit_mae": fit_mae,
    }


def judge_loss(H, w_raw, eta_free, z_l):
    """
    Compute model-side reconstruction loss.

    The weighted layer logits are first converted into a model-side distribution,
    which is then reconstructed by the ordered-threshold model.
    """
    w = torch.softmax(w_raw, dim=0)

    fused_logits = (H * w.view(1, -1, 1)).sum(dim=1)
    model_probs = torch.softmax(fused_logits, dim=1)

    eta = build_eta(eta_free)
    reconstructed_probs = ordinal_probs(eta, z_l)

    return torch.abs(reconstructed_probs - model_probs).sum()


def human_loss_initial(y_h, alpha_free, z_l, beta_raw, lambda_ord=0.0):
    """
    Compute hard-label human-side loss.

    This function is kept for compatibility with earlier experiments.
    """
    alpha = build_alpha(alpha_free)
    beta = F.softplus(beta_raw) + 1e-6

    z_h = z_l / beta
    pred_probs = ordinal_probs(alpha, z_h)

    loss_nll = F.nll_loss(torch.log(pred_probs), y_h, reduction="sum")

    levels = torch.arange(
        pred_probs.size(1),
        dtype=pred_probs.dtype,
        device=pred_probs.device,
    )
    pred_mean = (pred_probs * levels.view(1, -1)).sum(dim=1)
    loss_ord = ((pred_mean - y_h.float()) ** 2).sum()

    return loss_nll + lambda_ord * loss_ord


def human_loss(target_probs, alpha_free, z_l, beta_raw, lambda_ord=0.0, ord_type=None):
    """
    Compute human-side distributional loss.

    The main loss is soft cross-entropy between predicted human-side
    distributions and empirical human rating distributions. Optional ordinal
    regularization can be added through EMD or expected-score MSE.
    """
    alpha = build_alpha(alpha_free)
    beta = F.softplus(beta_raw) + 1e-6

    z_h = z_l / beta
    pred_probs = ordinal_probs(alpha, z_h)

    loss_soft_ce = -(target_probs * torch.log(pred_probs)).sum(dim=1).mean()

    if lambda_ord == 0.0 or ord_type is None:
        loss_ord = torch.tensor(0.0, device=pred_probs.device, dtype=pred_probs.dtype)
    elif ord_type == "emd":
        cdf_pred = torch.cumsum(pred_probs, dim=1)
        cdf_target = torch.cumsum(target_probs, dim=1)
        loss_ord = torch.abs(cdf_pred - cdf_target).sum(dim=1).mean()
    elif ord_type == "mean_mse":
        levels = torch.arange(
            pred_probs.size(1),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )
        pred_mean = (pred_probs * levels.view(1, -1)).sum(dim=1)
        target_mean = (target_probs * levels.view(1, -1)).sum(dim=1)
        loss_ord = ((pred_mean - target_mean) ** 2).mean()
    else:
        raise ValueError(f"Unsupported ord_type: {ord_type}")

    return loss_soft_ce + lambda_ord * loss_ord


def train_bridge_alternating(
    H,
    target_probs,
    num_outer_iters=50,
    num_steps_A=100,
    num_steps_B=100,
    num_steps_C=100,
    lr_A=1e-2,
    lr_B=1e-2,
    lr_C=1e-2,
    lambda_judge=1.0,
    lambda_human=1.0,
    lambda_ord=0.0,
    ord_type=None,
    w_init=None,
    eta_init=None,
    z_init=None,
):
    """
    Train Int-Rule++ with alternating optimization.

    The default setting follows the implementation details in the paper:
    50 outer iterations, 100 update steps for each stage, and learning rate 1e-2.

    Step A updates model-side thresholds and latent scores.
    Step B updates human-side thresholds and scale.
    Step C updates layer weights, model-side thresholds, and latent scores.
    """
    num_samples, num_layers, num_classes = H.shape
    device = H.device

    if w_init is None:
        w_raw = Parameter(torch.zeros(num_layers, device=device))
    else:
        w_init = w_init.to(device).float()
        w_raw = Parameter(torch.log(w_init + 1e-12))

    if eta_init is None:
        eta_free = Parameter(torch.zeros(num_classes - 2, device=device))
    else:
        eta_init = eta_init.to(device).float()
        gaps = torch.diff(eta_init).clamp_min(1e-6)
        eta_free = Parameter(torch.log(torch.expm1(gaps)).to(device))

    if z_init is None:
        z_l = Parameter(torch.zeros(num_samples, device=device))
    else:
        z_l = Parameter(z_init.to(device).float().clone())

    alpha_free = Parameter(torch.zeros(num_classes - 1, device=device))
    beta_raw = Parameter(torch.tensor(0.0, device=device))

    opt_A = Adam([eta_free, z_l], lr=lr_A)
    opt_B = Adam([alpha_free, beta_raw], lr=lr_B)
    opt_C = Adam([w_raw, eta_free, z_l], lr=lr_C)

    w = torch.softmax(w_raw, dim=0)
    eta = build_eta(eta_free)
    alpha = build_alpha(alpha_free)
    beta = F.softplus(beta_raw) + 1e-6

    for _ in range(num_outer_iters):
        for _ in range(num_steps_A):
            opt_A.zero_grad()
            loss_A = judge_loss(
                H=H,
                w_raw=w_raw.detach(),
                eta_free=eta_free,
                z_l=z_l,
            )
            loss_A.backward()
            opt_A.step()

        for _ in range(num_steps_B):
            opt_B.zero_grad()
            loss_B = human_loss(
                target_probs=target_probs,
                alpha_free=alpha_free,
                z_l=z_l.detach(),
                beta_raw=beta_raw,
                lambda_ord=lambda_ord,
                ord_type=ord_type,
            )
            loss_B.backward()
            opt_B.step()

        for _ in range(num_steps_C):
            opt_C.zero_grad()

            loss_j = judge_loss(
                H=H,
                w_raw=w_raw,
                eta_free=eta_free,
                z_l=z_l,
            )
            loss_h = human_loss(
                target_probs=target_probs,
                alpha_free=alpha_free.detach(),
                z_l=z_l,
                beta_raw=beta_raw.detach(),
                lambda_ord=lambda_ord,
                ord_type=ord_type,
            )
            loss_C = lambda_judge * loss_j + lambda_human * loss_h

            loss_C.backward()
            opt_C.step()

        with torch.no_grad():
            w = torch.softmax(w_raw, dim=0)
            eta = build_eta(eta_free)
            alpha = build_alpha(alpha_free)
            beta = F.softplus(beta_raw) + 1e-6

    return {
        "w": w.detach(),
        "eta": eta.detach(),
        "z_l": z_l.detach(),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
    }


def train_bridge_from_probs(
    P,
    target_probs,
    num_outer_iters=50,
    num_steps_A=100,
    num_steps_B=100,
    lr_A=1e-2,
    lr_B=1e-2,
    lambda_ord=0.0,
    ord_type=None,
    eta_init=None,
    z_init=None,
    verbose=True,
):
    """
    Train Final-Rule++ from a given probability distribution.

    The default setting uses 50 outer iterations, 100 update steps per stage,
    and learning rate 1e-2, consistent with the paper.
    """
    device = P.device
    P = P.float()
    target_probs = target_probs.to(device).float()

    num_samples, num_classes = P.shape

    if eta_init is None:
        eta_free = Parameter(torch.zeros(num_classes - 2, device=device))
    else:
        eta_init = eta_init.to(device).float()
        gaps = torch.diff(eta_init).clamp_min(1e-6)
        eta_free = Parameter(torch.log(torch.expm1(gaps)).to(device))

    if z_init is None:
        z_l = Parameter(torch.zeros(num_samples, device=device))
    else:
        z_l = Parameter(z_init.to(device).float().clone())

    alpha_free = Parameter(torch.zeros(num_classes - 1, device=device))
    beta_raw = Parameter(torch.tensor(0.0, device=device))

    opt_A = Adam([eta_free, z_l], lr=lr_A)
    opt_B = Adam([alpha_free, beta_raw], lr=lr_B)

    for outer in range(num_outer_iters):
        for _ in range(num_steps_A):
            opt_A.zero_grad()

            eta = build_eta(eta_free)
            model_probs = ordinal_probs(eta, z_l)
            loss_A = F.smooth_l1_loss(model_probs, P)

            loss_A.backward()
            opt_A.step()

        for _ in range(num_steps_B):
            opt_B.zero_grad()

            loss_B = human_loss(
                target_probs=target_probs,
                alpha_free=alpha_free,
                z_l=z_l.detach(),
                beta_raw=beta_raw,
                lambda_ord=lambda_ord,
                ord_type=ord_type,
            )

            loss_B.backward()
            opt_B.step()

        if verbose:
            with torch.no_grad():
                eta = build_eta(eta_free)
                alpha = build_alpha(alpha_free)
                beta = F.softplus(beta_raw) + 1e-6
                model_probs = ordinal_probs(eta, z_l)
                recon_mae = torch.abs(model_probs - P).mean()
                cur_human = human_loss(
                    target_probs=target_probs,
                    alpha_free=alpha_free,
                    z_l=z_l,
                    beta_raw=beta_raw,
                    lambda_ord=lambda_ord,
                    ord_type=ord_type,
                )

                print(f"[Final-Rule++] outer={outer}")
                print(f"  model_recon_mae = {recon_mae.item():.6f}")
                print(f"  human_loss = {cur_human.item():.6f}")
                print(f"  alpha = {alpha}")
                print(f"  beta = {beta.item():.4f}")
                print("-" * 50)

    with torch.no_grad():
        eta = build_eta(eta_free)
        alpha = build_alpha(alpha_free)
        beta = F.softplus(beta_raw) + 1e-6
        w = torch.ones(1, device=device)

    return {
        "w": w.detach(),
        "eta": eta.detach(),
        "z_l": z_l.detach(),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
    }


def pred_human_from_probs(
    P,
    result,
    z_init=None,
    beta_smoothl1=1e-3,
    num_epochs=2000,
    lr=1e-2,
    tol_loss=1e-10,
    tol_grad=1e-10,
    scheduler_factor=0.5,
    patience=50,
    lbfgs_max_iter=200,
    lbfgs_lr=1.0,
    verbose=False,
):
    """
    Predict human-side distributions for Final-Rule++.

    The model-side latent score is first inferred from P, then mapped to the
    human-side latent scale.
    """
    device = P.device
    P = P.double()

    eta = result["eta"].to(device).double()
    alpha = result["alpha"].to(device).double()
    beta = result["beta"].to(device).double()

    num_samples, num_classes = P.shape

    if z_init is None:
        z_init = torch.zeros(num_samples, device=device, dtype=torch.float64)
    else:
        z_init = z_init.detach().to(device).double()

    z_l = nn.Parameter(z_init[:, None])

    optimizer = optim.Adam([z_l], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=patience,
    )

    for _ in range(num_epochs):
        optimizer.zero_grad()

        model_probs = ordinal_probs(eta.float(), z_l.squeeze(1).float()).double()
        loss = F.smooth_l1_loss(model_probs, P, beta=beta_smoothl1)

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        final_loss = torch.abs(model_probs - P).mean()
        grad_norm = z_l.grad.norm().item()

        if final_loss.item() < tol_loss or grad_norm < tol_grad:
            break

    lbfgs_optimizer = optim.LBFGS(
        [z_l],
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        model_probs = ordinal_probs(eta.float(), z_l.squeeze(1).float()).double()
        loss = F.smooth_l1_loss(model_probs, P, beta=beta_smoothl1)
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)

    with torch.no_grad():
        z_l_pred = z_l.squeeze(1)
        z_h = z_l_pred / beta

        pred_probs = ordinal_probs(alpha.float(), z_h.float()).double()
        pred_label = pred_probs.argmax(dim=1)

        levels = torch.arange(num_classes, dtype=pred_probs.dtype, device=device)
        pred_score = (pred_probs * levels.view(1, -1)).sum(dim=1)

        if verbose:
            model_probs = ordinal_probs(eta.float(), z_l_pred.float()).double()
            recon_mae = torch.abs(model_probs - P).mean().item()
            print("Final-Rule++ model-side reconstruction MAE:", recon_mae)

    return z_l_pred.float(), pred_probs.float(), pred_label, pred_score.float()


def train_bridge_from_last_probs_mean(
    P,
    target_probs,
    num_outer_iters=50,
    num_steps_B=100,
    lr_B=1e-2,
    lambda_ord=0.0,
    ord_type=None,
    verbose=True,
):
    """
    Train a Rule variant using the expected score of P as the latent score.

    This function is used for Final-Rule when P is the final-layer distribution,
    and for Int-Rule when P is the weighted internal distribution.

    The default setting uses 50 outer iterations, 100 update steps, and
    learning rate 1e-2.
    """
    device = P.device
    P = P.float()
    target_probs = target_probs.to(device).float()

    num_samples, num_classes = P.shape
    levels = torch.arange(num_classes, device=device, dtype=P.dtype)

    z_l = (P * levels.view(1, -1)).sum(dim=1).detach()

    alpha_free = Parameter(torch.zeros(num_classes - 1, device=device))
    beta_raw = Parameter(torch.tensor(0.0, device=device))

    opt_B = Adam([alpha_free, beta_raw], lr=lr_B)

    for outer in range(num_outer_iters):
        for _ in range(num_steps_B):
            opt_B.zero_grad()

            loss_B = human_loss(
                target_probs=target_probs,
                alpha_free=alpha_free,
                z_l=z_l,
                beta_raw=beta_raw,
                lambda_ord=lambda_ord,
                ord_type=ord_type,
            )

            loss_B.backward()
            opt_B.step()

        if verbose:
            with torch.no_grad():
                alpha = build_alpha(alpha_free)
                beta = F.softplus(beta_raw) + 1e-6
                pred_probs = ordinal_probs(alpha, z_l / beta)
                pred_label = pred_probs.argmax(dim=1)
                target_label = target_probs.argmax(dim=1)

                train_acc = (pred_label == target_label).float().mean()

                print(f"[Rule] outer={outer}")
                print(f"  human_loss = {loss_B.item():.6f}")
                print(f"  train_acc = {train_acc.item():.4f}")
                print(f"  latent_score mean/std = {z_l.mean().item():.4f} / {z_l.std().item():.4f}")
                print(f"  pred_label counts = {torch.bincount(pred_label.long(), minlength=num_classes)}")
                print(f"  target_label counts = {torch.bincount(target_label.long(), minlength=num_classes)}")
                print(f"  alpha = {alpha}")
                print(f"  beta = {beta.item():.4f}")
                print("-" * 50)

    with torch.no_grad():
        alpha = build_alpha(alpha_free)
        beta = F.softplus(beta_raw) + 1e-6

    return {
        "alpha": alpha.detach(),
        "beta": beta.detach(),
        "z_l": z_l.detach(),
    }


def pred_human_from_last_probs_mean(P, result, verbose=False):
    """
    Predict human-side distributions using the expected score of P.

    This function supports both Final-Rule and Int-Rule depending on the input P.
    """
    device = P.device
    P = P.float()

    num_samples, num_classes = P.shape
    levels = torch.arange(num_classes, device=device, dtype=P.dtype)

    z_l = (P * levels.view(1, -1)).sum(dim=1)

    alpha = result["alpha"].to(device).float()
    beta = result["beta"].to(device).float()

    z_h = z_l / beta
    pred_probs = ordinal_probs(alpha, z_h)
    pred_label = pred_probs.argmax(dim=1)
    pred_score = (pred_probs * levels.view(1, -1)).sum(dim=1)

    if verbose:
        print("[Rule eval]")
        print("  latent_score mean/std =", z_l.mean().item(), z_l.std().item())
        print("  pred_label counts =", torch.bincount(pred_label.long(), minlength=num_classes))
        print("  pred_probs mean =", pred_probs.mean(dim=0))
        print("  pred_score mean/std =", pred_score.mean().item(), pred_score.std().item())
        print("-" * 50)

    return z_l.detach(), pred_probs.detach(), pred_label.detach(), pred_score.detach()


def append_rows_to_csv(rows, save_csv_path):
    """Append metric rows to a CSV file."""
    if save_csv_path is None or len(rows) == 0:
        return

    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    df = pd.DataFrame(rows)
    write_header = not os.path.exists(save_csv_path)

    print("writing csv:", save_csv_path, "rows:", len(rows))
    df.to_csv(save_csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def calc_corr(pred_score, target_score, type_r="pearson"):
    """Compute Pearson or Spearman correlation."""
    if isinstance(pred_score, torch.Tensor):
        pred_score = pred_score.detach().cpu().view(-1).numpy()
    else:
        pred_score = np.asarray(pred_score).reshape(-1)

    if isinstance(target_score, torch.Tensor):
        target_score = target_score.detach().cpu().view(-1).numpy()
    else:
        target_score = np.asarray(target_score).reshape(-1)

    if type_r == "pearson":
        return pearsonr(pred_score, target_score)[0]
    if type_r == "spearman":
        return spearmanr(pred_score, target_score)[0]

    raise ValueError("type_r must be either 'pearson' or 'spearman'.")


def print_correlations(
    all_score_dict,
    target_score,
    score_names=None,
    save_csv_path=None,
    model_name=None,
    dataset_name=None,
    valid_name=None,
):
    """Print and optionally save correlation metrics."""
    metrics = ["pearson", "spearman"]

    if score_names is None:
        score_names = list(all_score_dict.keys())

    table = PrettyTable(["score_type"] + metrics)
    rows = []

    for score_name in score_names:
        pearson = round(calc_corr(all_score_dict[score_name], target_score, "pearson"), 3)
        spearman = round(calc_corr(all_score_dict[score_name], target_score, "spearman"), 3)

        table.add_row([score_name, pearson, spearman])

        rows.append({
            "metric_group": "correlation",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "valid_name": valid_name,
            "score_type": score_name,
            "pearson": pearson,
            "spearman": spearman,
        })

    print(table)
    append_rows_to_csv(rows, save_csv_path)


def calibration_curves(target_label, y_probs, n_bins=10, strategy="quantile"):
    """Compute one-vs-rest calibration curves and the average calibration error."""
    if isinstance(target_label, torch.Tensor):
        target_label = target_label.detach().cpu().view(-1).numpy().astype(int)
    else:
        target_label = np.asarray(target_label).reshape(-1).astype(int)

    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    num_classes = y_probs.shape[1]
    probs_true = []
    probs_pred = []

    for k in range(num_classes):
        prob_true, prob_pred = calibration_curve(
            (target_label == k).astype(int),
            y_probs[:, k],
            n_bins=n_bins,
            strategy=strategy,
        )
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)

    valid_errors = [
        np.abs(true - pred).mean()
        for true, pred in zip(probs_true, probs_pred)
        if len(true) > 0
    ]

    error = float(np.mean(valid_errors)) if len(valid_errors) > 0 else np.nan
    return error, probs_true, probs_pred


def soft_calibration_curves(target_probs, y_probs, n_bins=10, strategy="quantile"):
    """
    Compute soft calibration against empirical human rating distributions.
    """
    if isinstance(target_probs, torch.Tensor):
        target_probs = target_probs.detach().cpu().numpy()
    else:
        target_probs = np.asarray(target_probs)

    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    num_classes = y_probs.shape[1]
    probs_true = []
    probs_pred = []

    for k in range(num_classes):
        pred_k = y_probs[:, k]
        true_k = target_probs[:, k]

        if strategy == "quantile":
            bin_edges = np.quantile(pred_k, np.linspace(0, 1, n_bins + 1))
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        true_bin = []
        pred_bin = []

        for i in range(n_bins):
            left, right = bin_edges[i], bin_edges[i + 1]

            if i == n_bins - 1:
                mask = (pred_k >= left) & (pred_k <= right)
            else:
                mask = (pred_k >= left) & (pred_k < right)

            if mask.sum() == 0:
                continue

            pred_bin.append(pred_k[mask].mean())
            true_bin.append(true_k[mask].mean())

        probs_true.append(np.array(true_bin))
        probs_pred.append(np.array(pred_bin))

    valid_errors = [
        np.abs(true - pred).mean()
        for true, pred in zip(probs_true, probs_pred)
        if len(true) > 0
    ]

    error = float(np.mean(valid_errors)) if len(valid_errors) > 0 else np.nan
    return error, probs_true, probs_pred