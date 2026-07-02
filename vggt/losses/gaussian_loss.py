# vggt/losses/gaussian_loss.py
# ============================================================
# Gaussian losses:
#   - opacity loss  = BCE (against dedup mask) + entropy + per-frame balance
#   - optional xyz / scale losses (used only when ground-truth is provided)
# ============================================================

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from vggt.losses.dedup_mask import compute_dedup_mask

_EPS = 1e-6


# ─────────── xyz / scale losses ───────────

def chamfer_distance_loss(pc1, pc2):
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)
    dist_sq = diff.pow(2).sum(-1)
    return 0.5 * (dist_sq.min(dim=1).values.mean()
                  + dist_sq.min(dim=0).values.mean())


def xyz_loss(pred_xyz, gt_xyz):
    std = gt_xyz.std() + _EPS
    if pred_xyz.shape[0] == gt_xyz.shape[0]:
        return F.l1_loss(pred_xyz / std, gt_xyz / std)
    return chamfer_distance_loss(pred_xyz / std, gt_xyz / std)


def scale_loss(pred_scale, gt_scale):
    pred_log = torch.log(pred_scale.clamp(min=1e-8))
    gt_log = torch.log(gt_scale.clamp(min=1e-8))
    std = gt_log.std() + _EPS
    if pred_scale.shape == gt_scale.shape:
        return F.l1_loss(pred_log / std, gt_log / std)
    return F.l1_loss(pred_log.mean() / std, gt_log.mean() / std)


# ─────────── opacity loss trio ───────────

def opacity_bce_loss(pred_opacity, dedup_mask):
    """
    pred_opacity: (B, S, HW, 1) in [0, 1]
    dedup_mask:   (B, S, H, W) in {0, 1}
    """
    B, S, HW, _ = pred_opacity.shape
    target = dedup_mask.reshape(B, S, HW, 1)
    return F.binary_cross_entropy(
        pred_opacity.clamp(_EPS, 1 - _EPS), target,
    )


def opacity_entropy_loss(pred_opacity):
    a = pred_opacity.clamp(_EPS, 1 - _EPS)
    H = -(a * a.log() + (1 - a) * (1 - a).log())
    return H.mean()


def opacity_balance_loss(pred_opacity, k=20.0):
    """Prevent one frame from monopolising the surface: balance the per-frame
    (soft) surface count across frames."""
    B, S, HW, _ = pred_opacity.shape
    soft_surf = torch.sigmoid((pred_opacity - 0.5) * k).sum(dim=2).squeeze(-1)
    target = soft_surf.mean(dim=1, keepdim=True)
    return ((soft_surf - target) ** 2 / (target.pow(2) + _EPS)).mean()


# ─────────── combined ───────────

def gaussian_loss_v5(
    pred: Dict[str, torch.Tensor],
    gt: Optional[Dict[str, torch.Tensor]] = None,
    extra_inputs: Optional[Dict] = None,
    w_xyz: float = 1.0,
    w_scale: float = 1.0,
    w_opa_bce: float = 1.0,
    w_opa_entropy: float = 0.05,
    w_opa_balance: float = 0.5,
    dedup_method: str = 'image',
) -> Dict[str, torch.Tensor]:
    """
    Args:
        pred:  GaussianHead output {xyz, rotation, scale, opacity, color}
        gt:    optional ground-truth (for xyz/scale losses). Opacity uses no GT.
        extra_inputs: must contain {xyz_base, depth_conf, extrinsics, intrinsics, H, W}
    """
    assert extra_inputs is not None, \
        "extra_inputs required (xyz_base/depth_conf/extr/intr/H/W)"

    losses = {}
    device = pred['xyz'].device

    # ─── xyz / scale losses (only if GT given) ───
    if gt is not None and 'xyz' in gt:
        losses['xyz'] = xyz_loss(pred['xyz'], gt['xyz'])
    else:
        losses['xyz'] = torch.tensor(0.0, device=device)

    if gt is not None and 'scale' in gt:
        losses['scale'] = scale_loss(pred['scale'], gt['scale'])
    else:
        losses['scale'] = torch.tensor(0.0, device=device)

    # ─── dedup mask ───
    dedup_mask = compute_dedup_mask(
        method=dedup_method,
        xyz_base=extra_inputs['xyz_base'],
        depth_conf=extra_inputs['depth_conf'],
        extrinsics=extra_inputs['extrinsics'],
        intrinsics=extra_inputs['intrinsics'],
        H=extra_inputs['H'],
        W=extra_inputs['W'],
    )

    # ─── opacity trio ───
    losses['opa_bce'] = opacity_bce_loss(pred['opacity'], dedup_mask)
    losses['opa_entropy'] = opacity_entropy_loss(pred['opacity'])
    losses['opa_balance'] = opacity_balance_loss(pred['opacity'])

    total = (w_xyz * losses['xyz']
             + w_scale * losses['scale']
             + w_opa_bce * losses['opa_bce']
             + w_opa_entropy * losses['opa_entropy']
             + w_opa_balance * losses['opa_balance'])
    losses['total'] = total

    # ─── monitoring metrics ───
    with torch.no_grad():
        losses['mask_mean'] = dedup_mask.mean()
        per_frame = dedup_mask.mean(dim=(0, 2, 3))  # (S,)
        losses['mask_per_frame'] = per_frame
        losses['mask_frame_std'] = per_frame.std()
        opa = pred['opacity'].detach()
        losses['opa_surface_ratio'] = (opa > 0.5).float().mean()
        losses['opa_fog_ratio'] = (opa < 0.1).float().mean()

    return losses