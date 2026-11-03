"""Ordered-threshold reconstruction and CDF calibration for RULE."""

import torch
import torch.nn.functional as F

from utils_compare import build_alpha, build_eta, model_reconstruction_kl, ordinal_probs


def calibrated_probs(probs, eta, alpha, beta):
    """Apply the instance CDF in equations (10)-(12) to ordered boundaries."""
    # Double precision keeps sigmoid coordinates distinct for a wider range.
    dtype = probs.dtype
    p, eta, boundaries = probs.double(), eta.double(), (beta * alpha).double()
    knots = torch.cat((eta.new_tensor([-float("inf")]), eta,
                       eta.new_tensor([float("inf")])))
    coords = torch.sigmoid(knots)
    x = torch.sigmoid(boundaries)
    bins = torch.searchsorted(knots[1:-1].contiguous(), boundaries.contiguous())
    left, right = coords[bins], coords[bins + 1]
    fraction = ((x - left) / (right - left).clamp_min(1e-15)).clamp(0, 1)
    cumulative = torch.cat((p.new_zeros((p.size(0), 1)), p.cumsum(dim=1)), dim=1)
    values = cumulative[:, bins] + p[:, bins] * fraction.unsqueeze(0)
    full = torch.cat((p.new_zeros((p.size(0), 1)), values,
                      p.new_ones((p.size(0), 1))), dim=1)
    out = torch.diff(full, dim=1).clamp_min(1e-12)
    return (out / out.sum(dim=1, keepdim=True)).to(dtype)


def _cross_entropy(target, predicted):
    return -(target * predicted.clamp_min(1e-12).log()).sum(dim=1).mean()


def train_cdf_rule(P, target_probs, H=None, w_init=None, plus=False,
                   num_outer_iters=50, num_steps_A=100, num_steps_B=100,
                   num_steps_C=100, lr_A=1e-2, lr_B=1e-2, lr_C=1e-2,
                   eta_init=None, z_init=None, verbose=False):
    """Fit final- or internal-distribution RULE; plus estimates latent scores."""
    P = P.detach().float()
    target_probs = target_probs.detach().to(P).float()
    n, k = P.shape
    device = P.device
    levels = torch.arange(k, dtype=P.dtype, device=device)
    if H is not None:
        H = H.detach().to(P)
        initial_w = (torch.ones(H.size(1), device=device) / H.size(1)
                     if w_init is None else w_init.detach().to(P))
        w_raw = torch.nn.Parameter(initial_w.clamp_min(1e-12).log())
    else:
        w_raw = None

    if plus:
        if eta_init is None:
            eta_free = torch.nn.Parameter(torch.zeros(k - 2, device=device))
        else:
            gaps = torch.diff(eta_init.detach().to(P)).clamp_min(1e-5)
            eta_free = torch.nn.Parameter(torch.log(torch.expm1(gaps)))
        z = torch.nn.Parameter((P * levels).sum(1).detach() if z_init is None
                               else z_init.detach().to(P).clone())
        make_eta = lambda: build_eta(eta_free)
        model_params = [eta_free, z]
    else:
        eta_free = torch.nn.Parameter(torch.zeros(k - 1, device=device))
        make_eta = lambda: build_alpha(eta_free)
        model_params = [eta_free]
        z = None

    alpha_free = torch.nn.Parameter(torch.zeros(k - 1, device=device))
    beta_raw = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt_a = torch.optim.Adam(model_params, lr=lr_A)
    opt_b = torch.optim.Adam([alpha_free, beta_raw], lr=lr_B)
    opt_c = (torch.optim.Adam(model_params + [w_raw], lr=lr_C)
             if w_raw is not None else None)

    def model_probs():
        if w_raw is None:
            return P
        return torch.softmax((H * torch.softmax(w_raw, 0)[None, :, None]).sum(1), 1)

    def reconstruction(p):
        latent = z if plus else (p * levels).sum(1)
        return model_reconstruction_kl(p, ordinal_probs(make_eta(), latent))

    def human(p, eta, alpha, beta):
        return _cross_entropy(target_probs, calibrated_probs(p, eta, alpha, beta))

    for outer in range(num_outer_iters):
        for _ in range(num_steps_A):
            opt_a.zero_grad()
            reconstruction(model_probs().detach()).backward()
            opt_a.step()
        for _ in range(num_steps_B):
            opt_b.zero_grad()
            loss = human(model_probs().detach(), make_eta().detach(),
                         build_alpha(alpha_free), F.softplus(beta_raw) + 1e-6)
            loss.backward()
            opt_b.step()
        if opt_c is not None:
            for _ in range(num_steps_C):
                opt_c.zero_grad()
                p = model_probs()
                loss = reconstruction(p) + human(p, make_eta(),
                    build_alpha(alpha_free).detach(), (F.softplus(beta_raw) + 1e-6).detach())
                loss.backward()
                opt_c.step()
        if verbose:
            print(f"[CDF-Rule{'++' if plus else ''}] outer={outer} human_ce={loss.item():.6f}")

    with torch.no_grad():
        result = {"eta": make_eta().detach(), "alpha": build_alpha(alpha_free).detach(),
                  "beta": (F.softplus(beta_raw) + 1e-6).detach()}
        if w_raw is not None:
            result["w"] = torch.softmax(w_raw, 0).detach()
        if plus:
            result["z_l"] = z.detach()
    return result


def predict_cdf_rule(P, result):
    """Predict without fitting any parameters on evaluation instances."""
    with torch.no_grad():
        probs = calibrated_probs(P, result["eta"], result["alpha"], result["beta"])
        levels = torch.arange(P.size(1), device=P.device, dtype=P.dtype)
        return probs, probs.argmax(1), (probs * levels).sum(1)


def train_bridge_alternating(H, target_probs, w_init=None, **kwargs):
    w = w_init.to(H)
    P = torch.softmax((H * w[None, :, None]).sum(1), 1)
    allowed = {key: kwargs[key] for key in (
        "num_outer_iters", "num_steps_A", "num_steps_B", "num_steps_C",
        "lr_A", "lr_B", "lr_C", "eta_init", "z_init") if key in kwargs}
    return train_cdf_rule(P, target_probs, H=H, w_init=w, plus=True, **allowed)


def train_bridge_from_probs(P, target_probs, **kwargs):
    allowed = {key: kwargs[key] for key in (
        "num_outer_iters", "num_steps_A", "num_steps_B", "lr_A", "lr_B",
        "eta_init", "z_init", "verbose") if key in kwargs}
    return train_cdf_rule(P, target_probs, plus=True, **allowed)


def train_bridge_from_last_probs_mean(P, target_probs, **kwargs):
    allowed = {key: kwargs[key] for key in (
        "num_outer_iters", "num_steps_B", "lr_B", "verbose") if key in kwargs}
    return train_cdf_rule(P, target_probs, **allowed)


def train_bridge_from_last_probs_mean_joint(H, target_probs, w_init, **kwargs):
    w = w_init.to(H)
    P = torch.softmax((H * w[None, :, None]).sum(1), 1)
    allowed = {key: kwargs[key] for key in (
        "num_outer_iters", "num_steps_B", "num_steps_C", "lr_B", "lr_C", "verbose")
        if key in kwargs}
    return train_cdf_rule(P, target_probs, H=H, w_init=w, **allowed)


def pred_human_from_probs(P, result, **kwargs):
    probs, labels, scores = predict_cdf_rule(P, result)
    return None, probs, labels, scores


def pred_human(H, result, **kwargs):
    P = torch.softmax((H * result["w"].to(H)[None, :, None]).sum(1), 1)
    return pred_human_from_probs(P, result)


pred_human_from_last_probs_mean = pred_human_from_probs
