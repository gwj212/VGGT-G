
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _tensor_to_uint8(img: torch.Tensor) -> "np.ndarray":
    import numpy as np
    img = img.detach().cpu().clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype("uint8")


def save_image(img_tensor: torch.Tensor, path: str) -> None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")
    arr = _tensor_to_uint8(img_tensor)
    Image.fromarray(arr).save(path)


def save_comparison(
    images: List[torch.Tensor],
    labels: List[str],
    path: str,
    psnr_values: Optional[List[Optional[float]]] = None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")
    n = len(images)
    arrays = [_tensor_to_uint8(img) for img in images]
    H, W = arrays[0].shape[:2]
    label_h = 28
    gap     = 4
    total_w = W * n + gap * (n - 1)
    total_h = H + label_h
    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for idx, (arr, label) in enumerate(zip(arrays, labels)):
        x_off = idx * (W + gap)
        canvas.paste(Image.fromarray(arr), (x_off, 0))
        psnr_str = ""
        if psnr_values is not None and psnr_values[idx] is not None:
            psnr_str = f"  PSNR={psnr_values[idx]:.2f}dB"
        text = label + psnr_str
        draw.text((x_off + 4, H + 6), text, fill=(220, 220, 220), font=font)
    canvas.save(path)


def _compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return 10 * math.log10(1.0 / (mse + 1e-10))


# ---------------------------------------------------------------------------
# Differentiable Gaussian rasterization
# ---------------------------------------------------------------------------

def render_gaussians(
    xyz: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    try:
        from gsplat import rasterization
    except ImportError:
        raise ImportError("gsplat not installed: pip install gsplat")
    device = xyz.device
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    viewmat = torch.eye(4, device=device, dtype=xyz.dtype)
    viewmat[:3, :3] = R
    viewmat[:3, 3]  = t
    viewmat = viewmat.unsqueeze(0)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    Ks = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        device=device, dtype=xyz.dtype
    ).unsqueeze(0)
    rendered, _, _ = rasterization(
        means=xyz,
        quats=rotation,
        scales=scale,
        opacities=opacity.squeeze(-1),
        colors=color,
        viewmats=viewmat,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1e10,
    )
    return rendered[0].permute(2, 0, 1).clamp(0, 1)


# ---------------------------------------------------------------------------
# Adaptive scale init from KNN nearest-neighbour distance
# ---------------------------------------------------------------------------

def _estimate_initial_scale(xyz: torch.Tensor, k: int = 4) -> torch.Tensor:
    N = xyz.shape[0]
    if N > 50000:
        idx = torch.randperm(N, device=xyz.device)[:10000]
        sample = xyz[idx]
    else:
        sample = xyz

    with torch.no_grad():
        batch_size = min(5000, sample.shape[0])
        all_dists = []
        for i in range(0, sample.shape[0], batch_size):
            batch = sample[i:i+batch_size]
            dist = torch.cdist(batch, sample)
            for j in range(batch.shape[0]):
                global_idx = i + j
                if global_idx < sample.shape[0]:
                    dist[j, global_idx] = float('inf')
            actual_k = min(k, dist.shape[1] - 1)
            topk_dist = dist.topk(actual_k, dim=-1, largest=False).values
            avg_dist = topk_dist.mean(dim=-1)
            all_dists.append(avg_dist)

        all_dists = torch.cat(all_dists)
        median_dist = all_dists.median().item()

    init_scale_val = max(median_dist * 0.5, 1e-4)
    log_scale = math.log(init_scale_val)
    if log_scale < -10.0:
        log_scale = -6.0
    if log_scale > 0.0:
        log_scale = -2.0
    return torch.full((N, 3), log_scale, device=xyz.device, dtype=xyz.dtype)


# ---------------------------------------------------------------------------
# Optimizable standard-3DGS parameter set
# ---------------------------------------------------------------------------

class StandardGaussians(nn.Module):
    def __init__(
        self,
        init_xyz: torch.Tensor,
        init_color: torch.Tensor,
        adaptive_scale: bool = True,
    ):
        super().__init__()
        N = init_xyz.shape[0]

        self.xyz = nn.Parameter(init_xyz.clone().float())

        color_logit = torch.logit(
            init_color.clone().float().clamp(1e-6, 1.0 - 1e-6)
        )
        self.color = nn.Parameter(color_logit)

        init_rot = torch.zeros(N, 4)
        init_rot[:, 0] = 1.0
        self.rotation_raw = nn.Parameter(init_rot)

        if adaptive_scale:
            init_scale = _estimate_initial_scale(init_xyz)
        else:
            init_scale = torch.full((N, 3), math.log(0.01))
        self.scale_raw = nn.Parameter(init_scale.to(init_xyz.device))

        init_opacity = torch.full((N, 1), 1.386)
        self.opacity_raw = nn.Parameter(init_opacity)

    @property
    def rotation(self):
        return F.normalize(self.rotation_raw, dim=-1)

    @property
    def scale(self):
        return torch.exp(self.scale_raw.clamp(-10.0, 4.0))

    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_raw)

    def get_gaussians_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'xyz':      self.xyz,
            'rotation': self.rotation,
            'scale':    self.scale,
            'opacity':  self.opacity,
            'color':    torch.sigmoid(self.color),
        }

    def prune_low_opacity(self, threshold: float = 0.01):
        with torch.no_grad():
            opacity_vals = torch.sigmoid(self.opacity_raw.squeeze(-1))
            mask = opacity_vals > threshold
            n_before = self.xyz.shape[0]
            n_keep = mask.sum().item()
            if n_keep < n_before and n_keep > 100:
                self.xyz.data = self.xyz.data[mask]
                self.color.data = self.color.data[mask]
                self.rotation_raw.data = self.rotation_raw.data[mask]
                self.scale_raw.data = self.scale_raw.data[mask]
                self.opacity_raw.data = self.opacity_raw.data[mask]
                return n_before - n_keep
        return 0


# ---------------------------------------------------------------------------
# Opacity binarization regularizer
# ---------------------------------------------------------------------------

def opacity_binary_regularization(opacity: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    o = opacity.clamp(eps, 1.0 - eps)
    entropy = -(o * torch.log(o) + (1.0 - o) * torch.log(1.0 - o))
    return entropy.mean()


# ---------------------------------------------------------------------------
# Single-view optimization
# ---------------------------------------------------------------------------

def optimize_gaussians(
    init_xyz: torch.Tensor,
    init_color: torch.Tensor,
    gt_image: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    H: int,
    W: int,
    num_iters: int = 7000,
    lr: float = 1e-3,
    verbose: bool = False,
    patience: int = 300,
    w_binary_max: float = 0.05,
    binary_warmup: int = 500,
    prune_interval: int = 1000,
    prune_threshold: float = 0.01,
    vis_dir: Optional[str] = None,
    vis_interval: int = 500,
    vis_prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """Single-view reference Gaussian optimization."""
    device = init_xyz.device
    gs = StandardGaussians(init_xyz, init_color, adaptive_scale=True).to(device)
    gt = gt_image.to(device).float()

    if verbose:
        print(f"  [3DGS] init points: {init_xyz.shape[0]}")
        opa_init = gs.opacity.mean().item()
        print(f"  [3DGS] init opacity mean: {opa_init:.4f}")
        scale_init = gs.scale.mean().item()
        print(f"  [3DGS] init scale mean: {scale_init:.6f}")

    do_vis = vis_dir is not None
    if do_vis:
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
        p = vis_prefix
        gt_save_path = str(vis_path / f"{p}gt.png")
        save_image(gt.cpu(), gt_save_path)
        vis_frames: List[torch.Tensor] = []
        vis_labels: List[str]          = []
        vis_psnrs:  List[Optional[float]] = []

    if do_vis:
        with torch.no_grad():
            params_init = gs.get_gaussians_dict()
            try:
                rendered_init = render_gaussians(
                    xyz=params_init['xyz'], rotation=params_init['rotation'],
                    scale=params_init['scale'], opacity=params_init['opacity'],
                    color=params_init['color'],
                    extrinsics=extrinsics, intrinsics=intrinsics, H=H, W=W,
                )
                init_psnr = _compute_psnr(rendered_init, gt)
                save_image(rendered_init.cpu(), str(vis_path / f"{p}init.png"))
                vis_frames.append(rendered_init.cpu())
                vis_labels.append("Init")
                vis_psnrs.append(init_psnr)
            except Exception:
                pass

    def _build_optimizer():
        return torch.optim.Adam([
            {'params': [gs.xyz],            'lr': lr * 10, 'name': 'xyz'},
            {'params': [gs.color],          'lr': lr * 10, 'name': 'color'},
            {'params': [gs.rotation_raw],   'lr': lr * 1,  'name': 'rotation'},
            {'params': [gs.scale_raw],      'lr': lr * 5,  'name': 'scale'},
            {'params': [gs.opacity_raw],    'lr': lr * 50, 'name': 'opacity'},
        ], betas=(0.9, 0.999), eps=1e-15)

    optimizer = _build_optimizer()

    warmup_iters = 100

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, num_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss  = float('inf')
    no_improve = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for i in range(num_iters):
        optimizer.zero_grad()

        params = gs.get_gaussians_dict()
        try:
            rendered = render_gaussians(
                xyz=params['xyz'], rotation=params['rotation'],
                scale=params['scale'], opacity=params['opacity'],
                color=params['color'],
                extrinsics=extrinsics, intrinsics=intrinsics, H=H, W=W,
            )
        except Exception as e:
            raise RuntimeError(f"render failed (step {i}): {e}")

        loss_l1   = F.l1_loss(rendered, gt)
        loss_ssim = 1.0 - ssim(rendered.unsqueeze(0), gt.unsqueeze(0))

        if i < binary_warmup:
            w_binary = 0.0
        else:
            progress = min(1.0, (i - binary_warmup) / max(1, binary_warmup))
            w_binary = w_binary_max * progress

        loss_binary = opacity_binary_regularization(params['opacity'])
        loss = 0.8 * loss_l1 + 0.2 * loss_ssim + w_binary * loss_binary

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gs.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        render_loss_val = (0.8 * loss_l1 + 0.2 * loss_ssim).item()
        if render_loss_val < best_loss - 1e-5:
            best_loss  = render_loss_val
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in gs.get_gaussians_dict().items()}
        else:
            no_improve += 1

        if (i > 0) and (i % prune_interval == 0) and (i < num_iters - 500):
            n_pruned = gs.prune_low_opacity(prune_threshold)
            if n_pruned > 0:
                optimizer = _build_optimizer()
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                for _ in range(i + 1):
                    scheduler.step()
                if verbose:
                    print(f"  [3DGS] iter {i}: pruned {n_pruned} low-opacity points, "
                          f"remaining {gs.xyz.shape[0]}")

        if verbose and (i % 500 == 0 or i == num_iters - 1):
            with torch.no_grad():
                psnr = _compute_psnr(rendered, gt)
                opa = params['opacity'].detach()
                opa_mean = opa.mean().item()
                opa_std  = opa.std().item()
                opa_low  = (opa < 0.1).float().mean().item()
                opa_high = (opa > 0.9).float().mean().item()
            print(
                f"  [3DGS] iter {i:5d}/{num_iters} | "
                f"loss={loss_val:.4f} PSNR={psnr:.2f}dB | "
                f"opa: mean={opa_mean:.3f} std={opa_std:.3f} "
                f"<0.1={opa_low:.1%} >0.9={opa_high:.1%} | "
                f"w_bin={w_binary:.4f} pts={gs.xyz.shape[0]}"
            )

        if do_vis and (i > 0) and (i % vis_interval == 0):
            with torch.no_grad():
                mid_psnr = _compute_psnr(rendered, gt)
                save_image(rendered.detach().cpu(), str(vis_path / f"{p}iter_{i:04d}.png"))
                vis_frames.append(rendered.detach().cpu())
                vis_labels.append(f"Iter {i}")
                vis_psnrs.append(mid_psnr)

        if no_improve >= patience:
            if verbose:
                print(f"  [3DGS] early stop at step {i} (no improvement for {patience} steps)")
            break

    final_state = best_state if best_state is not None else {
        k: v.detach() for k, v in gs.get_gaussians_dict().items()
    }

    if verbose:
        opa = final_state['opacity']
        print(f"\n  [3DGS] === final opacity distribution ===")
        print(f"    mean={opa.mean().item():.4f}  std={opa.std().item():.4f}")
        print(f"    <0.1: {(opa < 0.1).float().mean().item():.1%}")
        print(f"    0.1~0.5: {((opa >= 0.1) & (opa < 0.5)).float().mean().item():.1%}")
        print(f"    0.5~0.9: {((opa >= 0.5) & (opa < 0.9)).float().mean().item():.1%}")
        print(f"    >0.9: {(opa > 0.9).float().mean().item():.1%}")
        print(f"    final points: {opa.shape[0]}")

    if do_vis:
        with torch.no_grad():
            try:
                final_rendered = render_gaussians(
                    xyz=final_state['xyz'], rotation=final_state['rotation'],
                    scale=final_state['scale'], opacity=final_state['opacity'],
                    color=final_state['color'],
                    extrinsics=extrinsics, intrinsics=intrinsics, H=H, W=W,
                )
                final_psnr = _compute_psnr(final_rendered, gt)
                save_image(final_rendered.cpu(), str(vis_path / f"{p}final_best.png"))
                cmp_frames = [gt.cpu()] + vis_frames + [final_rendered.cpu()]
                cmp_labels = ["GT (Reference)"] + vis_labels + ["Final Best"]
                cmp_psnrs  = [None] + vis_psnrs + [final_psnr]
                save_comparison(cmp_frames, cmp_labels,
                                str(vis_path / f"{p}comparison.png"),
                                psnr_values=cmp_psnrs)
            except Exception as e:
                if verbose:
                    print(f"  [vis] final render save failed: {e}")

    return final_state


# ---------------------------------------------------------------------------
# Multi-view joint optimization
# ---------------------------------------------------------------------------

def optimize_gaussians_multiview(
    init_xyz: torch.Tensor,
    init_color: torch.Tensor,
    gt_images: List[torch.Tensor],          # multiple GT images [(3,H,W), ...]
    extrinsics_list: List[torch.Tensor],    # matching extrinsics [(4,4), ...]
    intrinsics_list: List[torch.Tensor],    # matching intrinsics [(3,3), ...]
    H: int,
    W: int,
    num_iters: int = 5000,
    lr: float = 1e-3,
    verbose: bool = False,
    patience: int = 500,
    w_binary_max: float = 0.05,
    binary_warmup: int = 500,
    prune_interval: int = 1000,
    prune_threshold: float = 0.01,
    views_per_step: int = 0,    # views sampled per step, 0 = all
) -> Dict[str, torch.Tensor]:
    """Multi-view joint reference Gaussian optimization.

    Differences from the single-view version:
      - each step renders multiple views and averages the loss
      - views_per_step > 0 randomly samples a subset of views (saves memory)
      - early stopping is based on the average render loss over views
      - no visualization (kept simple; not needed for precompute)
    """
    device = init_xyz.device
    S = len(gt_images)
    assert S == len(extrinsics_list) == len(intrinsics_list), \
        f"view count mismatch: {S} images, {len(extrinsics_list)} ext, {len(intrinsics_list)} itr"

    gs = StandardGaussians(init_xyz, init_color, adaptive_scale=True).to(device)

    # move all GT to device
    gts = [img.to(device).float() for img in gt_images]
    exts = [e.to(device).float() for e in extrinsics_list]
    itrs = [i.to(device).float() for i in intrinsics_list]

    if verbose:
        print(f"  [3DGS MV] init points: {init_xyz.shape[0]}, views: {S}")
        print(f"  [3DGS MV] views_per_step: {'all' if views_per_step <= 0 else views_per_step}")

    def _build_optimizer():
        return torch.optim.Adam([
            {'params': [gs.xyz],            'lr': lr * 10, 'name': 'xyz'},
            {'params': [gs.color],          'lr': lr * 10, 'name': 'color'},
            {'params': [gs.rotation_raw],   'lr': lr * 1,  'name': 'rotation'},
            {'params': [gs.scale_raw],      'lr': lr * 5,  'name': 'scale'},
            {'params': [gs.opacity_raw],    'lr': lr * 50, 'name': 'opacity'},
        ], betas=(0.9, 0.999), eps=1e-15)

    optimizer = _build_optimizer()

    warmup_iters = 100

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, num_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss  = float('inf')
    no_improve = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    use_all = (views_per_step <= 0 or views_per_step >= S)

    for i in range(num_iters):
        optimizer.zero_grad()

        params = gs.get_gaussians_dict()

        # ---- select views used this step ----
        if use_all:
            view_indices = list(range(S))
        else:
            view_indices = random.sample(range(S), views_per_step)

        total_l1 = 0.0
        total_ssim_loss = 0.0
        n_views = len(view_indices)

        for vi in view_indices:
            try:
                rendered = render_gaussians(
                    xyz=params['xyz'], rotation=params['rotation'],
                    scale=params['scale'], opacity=params['opacity'],
                    color=params['color'],
                    extrinsics=exts[vi], intrinsics=itrs[vi], H=H, W=W,
                )
            except Exception as e:
                raise RuntimeError(f"render failed (step {i}, view {vi}): {e}")

            total_l1 += F.l1_loss(rendered, gts[vi])
            total_ssim_loss += 1.0 - ssim(rendered.unsqueeze(0), gts[vi].unsqueeze(0))

        avg_l1 = total_l1 / n_views
        avg_ssim_loss = total_ssim_loss / n_views

        # binarization regularizer
        if i < binary_warmup:
            w_binary = 0.0
        else:
            progress = min(1.0, (i - binary_warmup) / max(1, binary_warmup))
            w_binary = w_binary_max * progress

        loss_binary = opacity_binary_regularization(params['opacity'])
        loss = 0.8 * avg_l1 + 0.2 * avg_ssim_loss + w_binary * loss_binary

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gs.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---- early stopping ----
        render_loss_val = (0.8 * avg_l1 + 0.2 * avg_ssim_loss).item()
        if render_loss_val < best_loss - 1e-5:
            best_loss  = render_loss_val
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in gs.get_gaussians_dict().items()}
        else:
            no_improve += 1

        # ---- pruning ----
        if (i > 0) and (i % prune_interval == 0) and (i < num_iters - 500):
            n_pruned = gs.prune_low_opacity(prune_threshold)
            if n_pruned > 0:
                optimizer = _build_optimizer()
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                for _ in range(i + 1):
                    scheduler.step()
                if verbose:
                    print(f"  [3DGS MV] iter {i}: pruned {n_pruned} points, "
                          f"remaining {gs.xyz.shape[0]}")

        # ---- logging ----
        if verbose and (i % 500 == 0 or i == num_iters - 1):
            with torch.no_grad():
                # PSNR over all views
                psnrs = []
                for vi in range(S):
                    r = render_gaussians(
                        params['xyz'].detach(), params['rotation'].detach(),
                        params['scale'].detach(), params['opacity'].detach(),
                        params['color'].detach(),
                        exts[vi], itrs[vi], H, W,
                    )
                    psnrs.append(_compute_psnr(r, gts[vi]))
                avg_psnr = sum(psnrs) / len(psnrs)
                opa = params['opacity'].detach()
            psnr_strs = [f"v{j}={p:.1f}" for j, p in enumerate(psnrs)]
            print(
                f"  [3DGS MV] iter {i:5d}/{num_iters} | "
                f"loss={loss.item():.4f} avgPSNR={avg_psnr:.2f}dB "
                f"[{' '.join(psnr_strs)}] | "
                f"opa: mean={opa.mean().item():.3f} std={opa.std().item():.3f} | "
                f"pts={gs.xyz.shape[0]}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  [3DGS MV] early stop at step {i}")
            break

    final_state = best_state if best_state is not None else {
        k: v.detach() for k, v in gs.get_gaussians_dict().items()
    }

    if verbose:
        opa = final_state['opacity']
        print(f"\n  [3DGS MV] === final opacity distribution ===")
        print(f"    mean={opa.mean().item():.4f}  std={opa.std().item():.4f}")
        print(f"    <0.1: {(opa < 0.1).float().mean().item():.1%}")
        print(f"    >0.9: {(opa > 0.9).float().mean().item():.1%}")
        print(f"    final points: {opa.shape[0]}")

    return final_state


# ---------------------------------------------------------------------------
# Simple SSIM
# ---------------------------------------------------------------------------

def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1, C2  = 0.01 ** 2, 0.03 ** 2
    channel = pred.shape[1]
    sigma  = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channel, 1, 1, 1)
    pad       = window_size // 2
    mu1       = F.conv2d(pred,        kernel, padding=pad, groups=channel)
    mu2       = F.conv2d(gt,          kernel, padding=pad, groups=channel)
    mu1_sq    = mu1 * mu1
    mu2_sq    = mu2 * mu2
    mu12      = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=channel) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()