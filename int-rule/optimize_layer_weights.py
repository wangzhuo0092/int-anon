import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from tqdm import trange


def set_seed(seed):
    """Set random seeds for reproducible optimization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_H_y(data_path, zero_based=True):
    """
    Load layer-wise candidate-label logits and hard human labels.

    This helper is kept for compatibility with earlier single-label experiments.

    Args:
        data_path: Path to the JSON result file.
        zero_based: Whether to convert labels from 1-based to 0-based.

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
        if "weighted_score" in sample and sample["weighted_score"] == -1:
            continue

        df = pd.DataFrame(sample["df"])
        logits = torch.tensor([x for x in df["logits"]], dtype=torch.float32)
        human_score = torch.tensor(sample["human_score"], dtype=torch.long)

        logits_list.append(logits)
        human_score_list.append(human_score)

    H = torch.stack(logits_list, dim=0)
    y_h = torch.stack(human_score_list, dim=0)

    if zero_based:
        y_h = y_h - 1

    return H, y_h


class CustomDataset(Dataset):
    """
    Dataset for learning layer weights in multi-layer judgment probing.

    Each sample contains:
        logits: layer-wise candidate-label logits, shape [L, K]
        target_probs: empirical human rating distribution, shape [K]
        target_mean: expected human score, scalar
        target_label: mode label, scalar
    """

    def __init__(self, data_path, label_start, drop_last_layer=True):
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        self.logits_list = []
        self.target_probs_list = []
        self.target_mean_list = []
        self.target_label_list = []

        for sample in all_data:
            if sample["weighted_socre"] == -1:
                continue

            if "human_score" in sample:
                if sample["human_score"] == -1:
                    continue
                if isinstance(sample["human_score"], list) and any(
                    score == -1 for score in sample["human_score"]
                ):
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

            if torch.any(scores < 0) or torch.any(scores >= num_classes):
                raise ValueError(
                    f"labels out of range: raw={sample['human_score']}, "
                    f"label_start={label_start}, shifted={scores.tolist()}, "
                    f"num_classes={num_classes}"
                )

            counts = torch.bincount(scores, minlength=num_classes).float()
            target_probs = counts / counts.sum()

            levels = torch.arange(num_classes, dtype=torch.float32)
            target_mean = (target_probs * levels).sum()
            target_label = counts.argmax()

            self.logits_list.append(logits)
            self.target_probs_list.append(target_probs)
            self.target_mean_list.append(target_mean)
            self.target_label_list.append(target_label)

        if len(self.logits_list) == 0:
            raise ValueError(f"No valid samples found in {data_path}")

    def __len__(self):
        return len(self.logits_list)

    def __getitem__(self, idx):
        return (
            self.logits_list[idx],
            self.target_probs_list[idx],
            self.target_mean_list[idx],
            self.target_label_list[idx],
        )


def optimize_layer_weights(
    data_path,
    label_start,
    num_epochs=2,
    lr=0.01,
    min_lr=1e-3,
    batch_size=8,
    seed=42,
):
    """
    Learn simplex layer weights for Int-Logit-W.

    The objective combines:
        1. soft cross-entropy against the human rating distribution;
        2. MSE between the predicted expected score and the human expected score.

    Args:
        data_path: Path to the calibration result file.
        label_start: Label offset. Use 0 for 0-based labels and 1 for 1-based labels.
        num_epochs: Number of optimization epochs.
        lr: Initial learning rate.
        min_lr: Minimum learning rate for the scheduler.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        layer_weights: Tensor with shape [L], normalized by softmax.
    """
    set_seed(seed)

    dataset = CustomDataset(
        data_path=data_path,
        label_start=label_start,
        drop_last_layer=True,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = dataset.logits_list[0].size(0)
    num_classes = dataset.logits_list[0].size(1)

    layer_weight_raw = torch.nn.Parameter(
        torch.zeros(num_layers, device=device),
        requires_grad=True,
    )
    loss_mix_raw = torch.nn.Parameter(
        torch.tensor([0.5], device=device),
        requires_grad=True,
    )

    optimizer = optim.Adam([layer_weight_raw, loss_mix_raw], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=min_lr,
    )

    score_values = torch.arange(num_classes, dtype=torch.float32, device=device)

    for _ in trange(num_epochs, desc="Learning layer weights"):
        total_loss = 0.0

        for (
            batch_logits,
            batch_target_probs,
            batch_target_mean,
            batch_target_label,
        ) in dataloader:
            batch_logits = batch_logits.to(device)
            batch_target_probs = batch_target_probs.to(device)
            batch_target_mean = batch_target_mean.to(device)

            loss_mix = torch.sigmoid(loss_mix_raw)
            layer_weights = torch.softmax(layer_weight_raw, dim=0)

            fused_logits = (batch_logits * layer_weights.view(1, -1, 1)).sum(dim=1)
            pred_probs = torch.softmax(fused_logits, dim=-1)
            pred_mean = (pred_probs * score_values.view(1, -1)).sum(dim=-1)

            loss_ce = -(
                batch_target_probs * torch.log(pred_probs.clamp_min(1e-12))
            ).sum(dim=1).mean()

            loss_mse = F.mse_loss(pred_mean, batch_target_mean)
            loss = loss_mix * loss_ce + (1.0 - loss_mix) * loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

    return torch.softmax(layer_weight_raw, dim=0).detach()


def GetLambdasZs(
    P,
    beta=1e-3,
    num_epochs=10000,
    lr=1e-2,
    tol_loss=1e-10,
    tol_grad=1e-10,
    scheduler_factor=0.999,
    patience=100,
    random_seed=42,
    verbose=True,
    lbfgs_max_iter=10000,
    lbfgs_lr=1.0,
):
    """
    Fit an ordinal reconstruction model to a probability matrix.

    This legacy helper estimates shared thresholds and one latent score per
    sample so that the induced ordered-threshold distribution reconstructs P.

    Args:
        P: Probability matrix with shape [N, K].
        beta: Smooth-L1 beta parameter.
        num_epochs: Number of Adam steps.
        lr: Adam learning rate.
        tol_loss: Early-stopping threshold for reconstruction MAE.
        tol_grad: Early-stopping threshold for gradient norm.
        scheduler_factor: Learning-rate decay factor.
        patience: Scheduler patience.
        random_seed: Random seed.
        verbose: Whether to print fitting diagnostics.
        lbfgs_max_iter: Maximum LBFGS refinement steps.
        lbfgs_lr: LBFGS learning rate.

    Returns:
        A dictionary with thresholds, latent scores, reconstruction loss, and gradient norm.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    scale = 0.1
    reg = 1e3
    num_thresholds = P.shape[1] - 1
    device = "cpu"

    P_bar = torch.tensor(P, dtype=torch.float64, device=device)

    lambdas = nn.Parameter(
        torch.tensor(
            np.sort(np.abs(np.random.normal(0, scale, num_thresholds - 1)))[None, :],
            device=device,
            dtype=torch.float64,
        )
    )
    latent_scores = nn.Parameter(
        torch.normal(
            0,
            scale / 1e3,
            size=(P_bar.shape[0], 1),
            dtype=torch.float64,
            device=device,
        )
    )

    optimizer = optim.Adam([lambdas, latent_scores], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=patience,
    )

    for _ in trange(num_epochs, desc="Fitting ordinal reconstruction"):
        optimizer.zero_grad()

        thresholds = torch.hstack(
            (
                torch.tensor([[0]], device=device, dtype=lambdas.dtype),
                lambdas,
            )
        )
        cdf = torch.sigmoid(thresholds - latent_scores)
        pred = torch.diff(
            torch.hstack(
                (
                    torch.zeros((cdf.shape[0], 1), device=device, dtype=torch.float64),
                    cdf,
                    torch.ones((cdf.shape[0], 1), device=device, dtype=torch.float64),
                )
            ),
            dim=1,
        )

        monotonic_penalty = ((torch.relu(-torch.diff(thresholds))) ** 2).mean()
        loss = F.smooth_l1_loss(pred, P_bar, beta=beta) + reg * monotonic_penalty

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        final_loss = torch.abs(pred - P_bar).mean()
        grad_norm = np.sqrt(
            lambdas.grad.norm().item() ** 2
            + latent_scores.grad.norm().item() ** 2
        )

        if final_loss < tol_loss or grad_norm < tol_grad:
            break

    lbfgs_optimizer = optim.LBFGS(
        [lambdas, latent_scores],
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs_optimizer.zero_grad()

        thresholds = torch.hstack(
            (
                torch.tensor([[0]], device=device, dtype=lambdas.dtype),
                lambdas,
            )
        )
        cdf = torch.sigmoid(thresholds - latent_scores)
        pred = torch.diff(
            torch.hstack(
                (
                    torch.zeros((cdf.shape[0], 1), device=device, dtype=torch.float64),
                    cdf,
                    torch.ones((cdf.shape[0], 1), device=device, dtype=torch.float64),
                )
            ),
            dim=1,
        )

        monotonic_penalty = ((torch.relu(-torch.diff(thresholds))) ** 2).mean()
        loss = F.smooth_l1_loss(pred, P_bar, beta=beta) + reg * monotonic_penalty
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)

    with torch.no_grad():
        thresholds_final = torch.hstack(
            (
                torch.tensor([[0]], device=device, dtype=lambdas.dtype),
                lambdas,
            )
        ).cpu().numpy().reshape(-1)

        latent_scores_final = latent_scores.cpu().numpy().reshape(-1)

        cdf = 1.0 / (
            1.0
            + np.exp(
                -(
                    thresholds_final[None, :]
                    - latent_scores_final[:, None]
                )
            )
        )
        pred = np.diff(
            np.hstack(
                (
                    np.zeros((cdf.shape[0], 1)),
                    cdf,
                    np.ones((cdf.shape[0], 1)),
                )
            ),
            axis=1,
        )

        final_loss = np.abs(pred - P_bar.cpu().numpy()).mean()
        grad_norm = np.sqrt(
            float(
                (
                    lambdas.grad.norm() ** 2
                    + latent_scores.grad.norm() ** 2
                ).cpu().item()
            )
        )

    if verbose:
        print("ordinal reconstruction MAE:", final_loss, "grad norm:", grad_norm)

    return {
        "lambdas": thresholds_final,
        "zs": latent_scores_final,
        "loss": final_loss,
        "grad_norm": grad_norm,
    }