#  Retained defaults:
#     - depth_unproject xyz_base, xyz_offset=0.3, warmup 500 / factor 0.1
#     - ColorHead enabled, dpt_feature_head unfrozen and added to the optimizer
#     - loss weights (W_RENDER=5.0, BCE=0.3, ENT=0.05, BAL=0.1)
#     - dedup_method='image'
#
#  Run (single GPU):
#     CUDA_VISIBLE_DEVICES=0 python train_nogt.py --single
#
#  Run (DDP, e.g. torchrun):
#     torchrun --nproc_per_node=N train_nogt.py
# ============================================================

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('VGGT_XYZ_BASE_SOURCE', 'depth_unproject')

import sys
import glob
import time
import math
import random
import datetime
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# Config  (set paths via env vars or edit the defaults below)
# ============================================================

# Directory of training scenes: one subdirectory per scene, each holding
# several multi-view images. REQUIRED — set TRAIN_DIR / TEST_DIR.
TRAIN_DIR         = os.environ.get("TRAIN_DIR", "")
TEST_DIR          = os.environ.get("TEST_DIR", "")
N_TEST_EVAL       = 20

EXP_TAG           = os.environ.get("EXP_TAG", "nogt")
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "output/nogt")
RENDER_DIR        = os.path.join(OUTPUT_DIR, "renders")
# Test cache only (lightweight, no GT). Separate dir to avoid clashing with
# any other cache.
TEST_CACHE_DIR    = os.path.join(OUTPUT_DIR, "test_cache")

NCCL_TIMEOUT_MINUTES = 20

GH_FRAMES_CHUNK_SIZE = 1
GH_USE_CHECKPOINT    = True

SCENE_MIN_VIEWS   = 2
SCENE_MAX_VIEWS   = 20
SCENE_NUM_VIEWS   = 0
TRAIN_VIEWS_PER_STEP = 5

# Upper bound on the training image long side (0 = unlimited, uses 518).
# Set 448/392 to lower the memory peak; images are downsampled to a multiple
# of 14 (VGGT patch=14). Override with env MAX_IMG_SIDE.
MAX_IMG_SIDE = int(os.environ.get('MAX_IMG_SIDE', '0'))

TEST_MIN_VIEWS    = 2
TEST_MAX_VIEWS    = 8
TEST_NUM_VIEWS    = 0

NUM_EPOCHS        = 10
LR                = 1e-4

# Resume: ''/unset = train from scratch; 'auto' = pick the latest step ckpt in
# OUTPUT_DIR; or a concrete .pth path. Resume restores model / optimizer /
# global_step / best metrics, and only overwrites the best ckpt on a higher
# PSNR / lower loss. Override with env RESUME.
RESUME_CHECKPOINT = os.environ.get('RESUME', '').strip()

ENABLE_COLOR_HEAD     = True
COLOR_LR_MULTIPLIER   = 8.0

FREEZE_DPT_FEATURE_HEAD = False
FEATURE_HEAD_LR_MULT    = 0.5

XYZ_OFFSET_LOG_SCALE_OVERRIDE = math.log(0.3)
WARMUP_STEPS  = 500
WARMUP_FACTOR = 0.1
XYZ_LR_MULTIPLIER = 0.5

# render / opacity loss weights
W_RENDER          = 5.0
W_RENDER_L1       = 0.7
W_RENDER_SSIM     = 0.3
W_OPA_BCE         = 0.3
W_OPA_ENTROPY     = 0.05
W_OPA_BALANCE     = 0.1

# geo-init cold-start anchor
GEO_SCALE_K   = float(os.environ.get('GEO_SCALE_K',   '1.0'))
GEO_OPA_CONST = float(os.environ.get('GEO_OPA_CONST', '0.5'))
W_GEO_SCALE   = float(os.environ.get('W_GEO_SCALE',   '1.0'))
W_GEO_OPA     = float(os.environ.get('W_GEO_OPA',     '0.5'))

DEDUP_METHOD              = 'image'
DEDUP_TIMEOUT_SEC         = 5.0
DEDUP_TIMING_LOG_EARLY    = 5
DEDUP_TIMING_LOG_INTERVAL = 200

PHASE2_START      = 5000
PHASE3_START      = 25000

LOG_INTERVAL      = 200
EVAL_INTERVAL     = 2000
SAVE_INTERVAL     = 10000

TEST_PATIENCE     = 20
MIN_TRAIN_STEPS   = 25000
HARD_STOP_STEP    = 41600   # total compute budget (global steps)

PSNR_SANE_MAX     = 55.0

# repo root: this file sits at the repo root (where `import vggt` works)
VGGT_ROOT = os.environ.get("VGGT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT, unproject_depth_to_world_torch
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.gs_optimizer import render_gaussians, ssim
from vggt.losses.dedup_mask import compute_dedup_mask


FORCE_SINGLE_GPU = (len(sys.argv) > 1 and sys.argv[1] == '--single')
if FORCE_SINGLE_GPU:
    for k in ['LOCAL_RANK', 'WORLD_SIZE', 'RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        os.environ.pop(k, None)


# ============================================================
# Logging: tee stdout/stderr into a .log file (all print/log_* are persisted)
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_file_logging(log_dir, tag):
    """Tee sys.stdout/stderr into <log_dir>/<tag>[_rankN].log (fixed name, so
       it is appended across runs). Print a RUN START banner at each launch."""
    rank = os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0'))
    os.makedirs(log_dir, exist_ok=True)
    suffix = '' if str(rank) == '0' else f'_rank{rank}'
    path = os.path.join(log_dir, f'{tag}{suffix}.log')      # fixed name -> resume appends
    fh = open(path, 'a', buffering=1)                       # append + line-buffered
    sys.stdout = _Tee(sys.__stdout__ or sys.stdout, fh)
    sys.stderr = _Tee(sys.__stderr__ or sys.stderr, fh)
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "#" * 70, flush=True)
    print(f"#  RUN START {ts}   (log appended, not overwritten)", flush=True)
    print("#" * 70, flush=True)
    print(f"[log] console output also appended to: {path}", flush=True)
    return path


# ============================================================
# Distributed
# ============================================================

def setup_ddp():
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl',
                                timeout=datetime.timedelta(minutes=NCCL_TIMEOUT_MINUTES))
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank(); world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log_main(rank, *a, **k):
    if rank == 0:
        print(*a, **k, flush=True)


def log_all(rank, *a, **k):
    print(f"[rank {rank}]", *a, **k, flush=True)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def sync_gradients_flat_masked(module, world_size, local_valid, device):
    if world_size <= 1 or not dist.is_initialized():
        return 1 if local_valid else 0
    valid_count = torch.tensor([1 if local_valid else 0], device=device, dtype=torch.int)
    dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)
    n_valid = int(valid_count.item())
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return n_valid
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        elif not local_valid:
            p.grad.zero_()
    flat = torch.cat([p.grad.detach().flatten() for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if n_valid > 0:
        flat.div_(n_valid)
    off = 0
    for p in params:
        n = p.grad.numel()
        p.grad.copy_(flat[off:off + n].view_as(p.grad)); off += n
    return n_valid


# ============================================================
# LPIPS / metrics / utils
# ============================================================

_lpips_fn = None

def get_lpips_fn(device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
            for p in _lpips_fn.parameters():
                p.requires_grad_(False)
        except ImportError:
            _lpips_fn = "unavailable"
    return _lpips_fn


def compute_lpips(pred, gt, device):
    fn = get_lpips_fn(device)
    if fn == "unavailable":
        return -1.0
    with torch.no_grad():
        p = pred.unsqueeze(0).float().to(device) * 2 - 1
        g = gt.unsqueeze(0).float().to(device) * 2 - 1
        return fn(p, g).item()


def copy_point_head_to_feature_head(model):
    assert model.point_head is not None and model.dpt_feature_head is not None
    ph, fh = model.point_head.state_dict(), model.dpt_feature_head.state_dict()
    copied = 0
    for name, param in ph.items():
        if name in fh and param.shape == fh[name].shape:
            fh[name].copy_(param); copied += 1
    model.dpt_feature_head.load_state_dict(fh)
    return copied


def find_images(directory):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(directory, '**', e), recursive=True))
    imgs.sort()
    return imgs


def find_scenes(directory, min_views=2, max_views=20, num_views=0):
    scene_dirs = sorted(d for d in glob.glob(os.path.join(directory, "*")) if os.path.isdir(d))
    scenes = []
    for sd in scene_dirs:
        imgs = find_images(sd)
        if len(imgs) < min_views:
            continue
        n = min(num_views, len(imgs)) if num_views > 0 else min(max_views, len(imgs))
        if n < len(imgs):
            idx = np.linspace(0, len(imgs) - 1, n, dtype=int)
            imgs = [imgs[i] for i in idx]
        scenes.append(imgs)
    return scenes


def format_duration(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    if h > 0: return f"{h}h {m}m {sec}s"
    if m > 0: return f"{m}m {sec}s"
    return f"{sec}s"


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    if mse < 1e-10 or math.isnan(mse):
        return 60.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_val(pred, gt, window_size=11):
    p = pred.float().unsqueeze(0); g = gt.float().unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    ch = p.shape[1]; sigma = 1.5
    coords = torch.arange(window_size, dtype=p.dtype, device=p.device) - window_size // 2
    k1d = torch.exp(-(coords**2) / (2 * sigma**2)); k1d /= k1d.sum()
    kernel = k1d.outer(k1d).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(p, kernel, padding=pad, groups=ch)
    mu2 = F.conv2d(g, kernel, padding=pad, groups=ch)
    m1sq, m2sq, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(p * p, kernel, padding=pad, groups=ch) - m1sq
    s2 = F.conv2d(g * g, kernel, padding=pad, groups=ch) - m2sq
    s12 = F.conv2d(p * g, kernel, padding=pad, groups=ch) - m12
    ssim_map = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1sq + m2sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def save_render_comparison(rendered, gt, step, psnr, ssim_val, save_dir, prefix="", lpips_val=None):
    os.makedirs(save_dir, exist_ok=True)
    def to_pil(t):
        arr = (t.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr.transpose(1, 2, 0))
    r, g_img = to_pil(rendered), to_pil(gt)
    W_img, H_img = r.width, r.height
    canvas = Image.new("RGB", (W_img * 2 + 4, H_img), (128, 128, 128))
    canvas.paste(r, (0, 0)); canvas.paste(g_img, (W_img + 4, 0))
    ls = f"_lpips{lpips_val:.4f}" if lpips_val is not None and lpips_val >= 0 else ""
    fname = f"{prefix}step{step:06d}_psnr{psnr:.2f}_ssim{ssim_val:.4f}{ls}.png"
    canvas.save(os.path.join(save_dir, fname))
    return fname


# ============================================================
# Loss (no GT param loss; only geo-init / render / opacity)
# ============================================================
_EPS = 1e-6


def geo_init_param_loss(pred_gaussians, xyz_base, S_sel, H, W,
                        scale_k, opa_const, w_scale, w_opa, strength):
    """GT-free cold-start anchor (pure geometry): scale -> scale_k * local point
       spacing (log space); opacity -> constant. Applied to the per-pixel
       Gaussians of all selected frames; strength = phase annealing. Targets
       are always detached."""
    scale = pred_gaussians['scale'][0, :S_sel]
    opa = pred_gaussians['opacity'][0, :S_sel]
    with torch.no_grad():
        xb = xyz_base[0, :S_sel].reshape(S_sel, H, W, 3).float()
        dr = (xb[:, :, 1:, :] - xb[:, :, :-1, :]).norm(dim=-1)
        dd = (xb[:, 1:, :, :] - xb[:, :-1, :, :]).norm(dim=-1)
        spacing = torch.zeros(S_sel, H, W, device=xb.device)
        cnt = torch.zeros(S_sel, H, W, device=xb.device)
        spacing[:, :, 1:] += dr; cnt[:, :, 1:] += 1
        spacing[:, :, :-1] += dr; cnt[:, :, :-1] += 1
        spacing[:, 1:, :] += dd; cnt[:, 1:, :] += 1
        spacing[:, :-1, :] += dd; cnt[:, :-1, :] += 1
        spacing = (spacing / cnt.clamp(min=1)).reshape(S_sel, H * W, 1).clamp(1e-6)
        target_log_scale = torch.log((scale_k * spacing).clamp(1e-8))
    l_scale = (torch.log(scale.clamp(1e-8)) - target_log_scale).abs().mean()
    l_opa = ((opa - opa_const) ** 2).mean()
    total = strength * (w_scale * l_scale + w_opa * l_opa)
    return total, float(l_scale.item()), float(l_opa.item())


def opacity_bce_loss(pred_opacity, dedup_mask):
    B, S, HW, _ = pred_opacity.shape
    target = dedup_mask.reshape(B, S, HW, 1)
    # Sanitize: in degenerate scenes dedup_mask may contain NaN/Inf/out-of-range
    # values; feeding BCE directly triggers a device-side assert (target must be
    # in [0,1]). nan_to_num + clamp first.
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred_opacity.clamp(_EPS, 1 - _EPS), target)


def opacity_entropy_loss(pred_opacity):
    a = pred_opacity.clamp(_EPS, 1 - _EPS)
    return (-(a * a.log() + (1 - a) * (1 - a).log())).mean()


def opacity_balance_loss(pred_opacity, k=20.0):
    soft_surf = torch.sigmoid((pred_opacity - 0.5) * k).sum(dim=2).squeeze(-1)
    target = soft_surf.mean(dim=1, keepdim=True)
    return ((soft_surf - target) ** 2 / (target.pow(2) + _EPS)).mean()


def compute_render_loss(pred_frame, gt_image, ext, itr, H, W):
    rendered = render_gaussians(
        xyz=pred_frame['xyz'].float(), rotation=pred_frame['rotation'].float(),
        scale=pred_frame['scale'].float(), opacity=pred_frame['opacity'].float(),
        color=pred_frame['color'].float(), extrinsics=ext, intrinsics=itr, H=H, W=W)
    l1 = F.l1_loss(rendered, gt_image)
    sloss = 1.0 - ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0))
    return W_RENDER_L1 * l1 + W_RENDER_SSIM * sloss, rendered


def build_param_groups(gh, base_lr, xyz_lr_mult=1.0, color_lr_mult=1.0):
    geo, xyz, color = [], [], []
    for name, p in gh.named_parameters():
        if not p.requires_grad:
            continue
        if 'color_head' in name:
            color.append(p)
        elif 'xyz_head' in name or 'xyz_offset' in name:
            xyz.append(p)
        else:
            geo.append(p)
    groups = [{'params': geo, 'lr': base_lr, 'name': 'geo'},
              {'params': xyz, 'lr': base_lr * xyz_lr_mult, 'name': 'xyz'}]
    if color:
        groups.append({'params': color, 'lr': base_lr * color_lr_mult, 'name': 'color'})
    return groups


def safe_compute_dedup_mask(method, xyz_base, depth_conf, extr, intr, H, W,
                            global_step, timeout_sec=DEDUP_TIMEOUT_SEC, log_timing=False, rank=0):
    t = time.time()
    try:
        mask = compute_dedup_mask(method=method, xyz_base=xyz_base, depth_conf=depth_conf,
                                  extrinsics=extr, intrinsics=intr, H=H, W=W)
        el = time.time() - t
        if log_timing:
            log_all(rank, f"  [mask] step{global_step} method={method} took={el:.2f}s ratio={mask.mean().item():.2f}")
        return mask, el, None
    except Exception as e:
        el = time.time() - t
        log_all(rank, f"  [mask] step{global_step} {method} failed: {e}, fall back to 'none'")
        B, S, _, _, _ = xyz_base.shape
        return torch.ones(B, S, H, W, device=xyz_base.device), el, str(e)


# ============================================================
# Test-set precompute (lightweight, no GT)
# ============================================================

def precompute_test_scene_light(model, image_paths, device, dtype, cache_path):
    if os.path.exists(cache_path):
        try:
            d = torch.load(cache_path, map_location='cpu')
            assert all(k in d for k in ('gt_images', 'images_input', 'exts', 'itrs', 'num_views'))
            return d
        except Exception:
            os.remove(cache_path)
    images = load_and_preprocess_images(image_paths).to(device)
    images_input = images.unsqueeze(0)
    S, _, H, W = images.shape
    model.eval()
    _gh, _dfh = model.gaussian_head, model.dpt_feature_head
    model.gaussian_head, model.dpt_feature_head = None, None
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(images_input)
    finally:
        model.gaussian_head, model.dpt_feature_head = _gh, _dfh
    extr, intr = pose_encoding_to_extri_intri(preds['pose_enc'], image_size_hw=(H, W))
    d = {'scene_name': Path(image_paths[0]).parent.name, 'image_paths': image_paths,
         'num_views': S, 'gt_images': [images[s].float().cpu() for s in range(S)],
         'exts': [extr[0, s].float().cpu() for s in range(S)],
         'itrs': [intr[0, s].float().cpu() for s in range(S)],
         'H': H, 'W': W, 'images_input': images_input.cpu()}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(d, cache_path)
    del preds, images_input, images
    torch.cuda.empty_cache()
    return d


def precompute_test_split(model, my_scenes, device, dtype, cache_dir, rank, idx_map):
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    for li, scene_imgs in enumerate(my_scenes):
        gi = idx_map[li]
        name = Path(scene_imgs[0]).parent.name
        cp = os.path.join(cache_dir, f"test_{gi:05d}_{name}.pt")
        try:
            precompute_test_scene_light(model, scene_imgs, device, dtype, cp)
        except Exception as e:
            log_all(rank, f"  [precomp test] {name} failed: {e}")
            torch.cuda.empty_cache()
    log_all(rank, f"  [precomp test] DONE: {len(my_scenes)} scenes | {format_duration(time.time()-t0)}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_single_test_scene(model, td, device, dtype, viz_idx=None):
    images_input = td['images_input'].to(device)
    S, H, W = td['num_views'], td['H'], td['W']
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pe = model(images_input)
        ps = {k: v[0].reshape(-1, v.shape[-1]).float() for k, v in pe['gaussians'].items()
              if isinstance(v, torch.Tensor) and v.dim() >= 4}
        psnrs, ssims, lpipss, viz = [], [], [], {}
        viz_set = set(viz_idx) if viz_idx else set()
        use_merged = True
        for s in range(S):
            ext = td['exts'][s].to(device); itr = td['itrs'][s].to(device); gt = td['gt_images'][s].to(device)
            try:
                if use_merged:
                    rd = render_gaussians(ps['xyz'], ps['rotation'], ps['scale'], ps['opacity'], ps['color'], ext, itr, H, W)
                else:
                    ef = {k: v[0, s].float() for k, v in pe['gaussians'].items() if isinstance(v, torch.Tensor) and v.dim() >= 4}
                    rd = render_gaussians(ef['xyz'], ef['rotation'], ef['scale'], ef['opacity'], ef['color'], ext, itr, H, W)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and use_merged:
                    torch.cuda.empty_cache(); use_merged = False
                    try:
                        ef = {k: v[0, s].float() for k, v in pe['gaussians'].items() if isinstance(v, torch.Tensor) and v.dim() >= 4}
                        rd = render_gaussians(ef['xyz'], ef['rotation'], ef['scale'], ef['opacity'], ef['color'], ext, itr, H, W)
                    except Exception:
                        torch.cuda.empty_cache(); continue
                else:
                    torch.cuda.empty_cache(); continue
            p_s = compute_psnr(rd, gt); s_s = compute_ssim_val(rd, gt); lp = compute_lpips(rd, gt, device)
            psnrs.append(p_s); ssims.append(s_s)
            if lp >= 0:
                lpipss.append(lp)
            if s in viz_set:
                viz[s] = (rd, gt, p_s, s_s, lp)
            torch.cuda.empty_cache()
    return (sum(psnrs) / max(len(psnrs), 1), sum(ssims) / max(len(ssims), 1),
            sum(lpipss) / len(lpipss) if lpipss else -1.0, viz)


def evaluate_on_test_set(model, tds, device, dtype, step, render_dir, save_renders=3):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    n_bad = 0
    for i, td in enumerate(tds):
        viz_views = sorted(set([0, td['num_views'] // 2, td['num_views'] - 1])) if i < save_renders else None
        try:
            p, s, lp, viz = evaluate_single_test_scene(model, td, device, dtype, viz_views)
            if not math.isfinite(p) or p > PSNR_SANE_MAX:
                n_bad += 1; continue
            psnrs.append(p); ssims.append(s)
            if lp >= 0:
                lpipss.append(lp)
            if viz_views and viz:
                for v in viz_views:
                    if v not in viz:
                        continue
                    rd, gt, pp, ss, ll = viz[v]
                    save_render_comparison(rd, gt, step, pp, ss, render_dir, prefix=f"scene{i:02d}_v{v}_", lpips_val=ll)
        except Exception:
            pass
        torch.cuda.empty_cache()
    if not psnrs:
        return 0.0, 0.0, -1.0, "eval failed"
    extra = f" (skipped bad {n_bad})" if n_bad else ""
    return (sum(psnrs) / len(psnrs), sum(ssims) / len(ssims),
            sum(lpipss) / len(lpipss) if lpipss else -1.0,
            f"avg over {len(psnrs)} scenes{extra} | PSNR [{min(psnrs):.1f}, {max(psnrs):.1f}]")


# ============================================================
# Sanity check (loads from raw images, no cache)
# ============================================================

def sanity_check_xyz_base(model, scene_imgs, device, dtype, rank):
    if rank != 0:
        return
    log_main(rank, "\n" + "=" * 70)
    log_main(rank, "  [Sanity Check] xyz_base + dedup_mask health check (from raw images)")
    log_main(rank, "=" * 70)
    try:
        n = min(5, len(scene_imgs))
        images = load_and_preprocess_images(scene_imgs[:n]).to(device)
        images_input = images.unsqueeze(0)
        _, _, H, W = images.shape
    except Exception as e:
        log_main(rank, f"  load failed: {e}"); return
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images_input)
    if not all(k in preds for k in ('world_points', 'depth', 'pose_enc')):
        log_main(rank, "  predictions incomplete, skip"); return
    old_xyz = preds['world_points'].float()
    extr, intr = pose_encoding_to_extri_intri(preds['pose_enc'], image_size_hw=(H, W))
    new_xyz = unproject_depth_to_world_torch(preds['depth'].float(), extr.float(), intr.float())
    rel = (old_xyz - new_xyz).norm(dim=-1).mean() / (old_xyz.norm(dim=-1).mean() + 1e-8)
    log_main(rank, f"  scene name        : {Path(scene_imgs[0]).parent.name}")
    log_main(rank, f"  relative diff     : {rel.item()*100:.1f}% of scene scale")
    try:
        xyz_base = unproject_depth_to_world_torch(preds['depth'].float(), extr.float(), intr.float())
        mask = compute_dedup_mask(method=DEDUP_METHOD, xyz_base=xyz_base,
                                  depth_conf=preds['depth_conf'].float(),
                                  extrinsics=extr.float(), intrinsics=intr.float(), H=H, W=W)
        pf = mask[0].mean(dim=(1, 2)).cpu().numpy()
        log_main(rank, f"  dedup mask ratio  : {mask.mean().item()*100:.1f}% | "
                       f"per-frame [{','.join(f'{x*100:.0f}%' for x in pf)}] fstd {float(pf.std())*100:.2f}%")
    except Exception as e:
        log_main(rank, f"  dedup_mask check failed: {e}")
    log_main(rank, "=" * 70 + "\n")
    del preds, old_xyz, new_xyz, images_input, images
    torch.cuda.empty_cache()


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, step, epoch, loss_val, path, metrics=None, best_state=None):
    ckpt = {'global_step': step, 'epoch': epoch, 'loss': loss_val,
            'gaussian_head_state_dict': model.gaussian_head.state_dict(),
            'dpt_feature_head_state_dict': model.dpt_feature_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
    if metrics is not None:
        ckpt['metrics'] = metrics
    if best_state is not None:                  # resume tracking: store current best metrics too
        ckpt['best_state'] = best_state
    torch.save(ckpt, path)


def _best_state(best_psnr, best_ssim, best_lpips, best_loss):
    return {'best_psnr': best_psnr, 'best_ssim': best_ssim,
            'best_lpips': best_lpips, 'best_loss': best_loss}


def find_resume_checkpoint(output_dir, tag):
    """When RESUME='auto', pick a checkpoint to resume: prefer the largest step;
       else best_loss; else best_test."""
    step_ckpts = glob.glob(os.path.join(output_dir, f"gaussian_head_step*_{tag}.pth"))
    if step_ckpts:
        def _stepnum(p):
            m = re.search(r'step(\d+)_', os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(step_ckpts, key=_stepnum)
    for name in (f"gaussian_head_best_loss_{tag}.pth", f"gaussian_head_best_test_{tag}.pth"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_checkpoint_for_resume(model, optimizer, ckpt_path, output_dir, tag, device, rank):
    """Load model + optimizer state; return
       (next_step, best_psnr, best_ssim, best_lpips, best_loss).
       - next_step = ckpt global_step + 1 (that step is done; continue from next).
       - best metrics come from ckpt['best_state'] (new format); for old ckpts
         without it, fall back to best_test ckpt['metrics'] and best_loss
         ckpt['loss'] in the same dir, so existing best checkpoints are not
         overwritten by the first eval after resume.
       - after loading optimizer state on CPU, move tensors to device (otherwise
         Adam raises a device-mismatch error)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.gaussian_head.load_state_dict(ckpt['gaussian_head_state_dict'])
    if 'dpt_feature_head_state_dict' in ckpt:
        model.dpt_feature_head.load_state_dict(ckpt['dpt_feature_head_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            log_main(rank, f"  optimizer state load failed ({e}), continue with a fresh optimizer")

    next_step = int(ckpt.get('global_step', 0)) + 1

    bs = ckpt.get('best_state', {})
    best_psnr  = float(bs.get('best_psnr', 0.0))
    best_ssim  = float(bs.get('best_ssim', 0.0))
    best_lpips = float(bs.get('best_lpips', float('inf')))
    best_loss  = float(bs.get('best_loss', float('inf')))

    if not bs:    # old ckpt: fall back to existing best_test / best_loss to protect them
        bt = os.path.join(output_dir, f"gaussian_head_best_test_{tag}.pth")
        if os.path.exists(bt):
            try:
                m = torch.load(bt, map_location='cpu').get('metrics', {})
                if 'psnr' in m:
                    best_psnr = max(best_psnr, float(m['psnr']))
                if 'ssim' in m:
                    best_ssim = float(m['ssim'])
                if m.get('lpips', -1) is not None and float(m.get('lpips', -1)) >= 0:
                    best_lpips = float(m['lpips'])
            except Exception:
                pass
        bl = os.path.join(output_dir, f"gaussian_head_best_loss_{tag}.pth")
        if os.path.exists(bl):
            try:
                lv = torch.load(bl, map_location='cpu').get('loss', None)
                if lv is not None:
                    best_loss = min(best_loss, float(lv))
            except Exception:
                pass

    for st in optimizer.state.values():          # move optimizer state tensors to GPU
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)

    return next_step, best_psnr, best_ssim, best_lpips, best_loss


# ============================================================
# Training data: load selected views per step from raw images (no cache, no GT)
# ============================================================

def _round14(x):
    return max(14, int(round(x / 14)) * 14)


def load_selected_views(scene_imgs, n_pick, device):
    """Randomly pick n_pick views and load only those images (avoids the cost of
       loading the whole set when H/W varies across steps in a long run).
       Returns (images_input[1,n,3,H,W] on device, gt_images_sel list, selected, H, W)."""
    S_total = len(scene_imgs)
    n = min(n_pick, S_total)
    selected = random.sample(range(S_total), n)
    sel_paths = [scene_imgs[s] for s in selected]
    imgs = load_and_preprocess_images(sel_paths)            # CPU [n,3,H,W]
    n, _, H, W = imgs.shape
    # optional downsample to reduce peak memory (keep multiple of 14; done on CPU)
    if MAX_IMG_SIDE > 0 and max(H, W) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(H, W)
        nH, nW = _round14(H * scale), _round14(W * scale)
        try:
            imgs = F.interpolate(imgs, size=(nH, nW), mode='bilinear',
                                 align_corners=False, antialias=True)
        except TypeError:                                   # older torch has no antialias
            imgs = F.interpolate(imgs, size=(nH, nW), mode='bilinear', align_corners=False)
        H, W = nH, nW
    imgs = imgs.to(device)
    images_input = imgs.unsqueeze(0)
    gt_images_sel = [imgs[i].float() for i in range(n)]
    return images_input, gt_images_sel, selected, H, W


# ============================================================
# Main
# ============================================================

def main():
    log_path = setup_file_logging(OUTPUT_DIR, EXP_TAG)
    total_start = time.time()
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(local_rank)[0] >= 8
             else torch.float16)

    PHASE2_LOCAL = max(1, PHASE2_START // world_size)
    PHASE3_LOCAL = max(1, PHASE3_START // world_size)
    MIN_TRAIN_LOCAL = max(1, MIN_TRAIN_STEPS // world_size)
    HARD_STOP_LOCAL = max(1, HARD_STOP_STEP // world_size)
    EVAL_LOCAL = max(1, EVAL_INTERVAL // world_size)
    SAVE_LOCAL = max(1, SAVE_INTERVAL // world_size)
    LOG_LOCAL = max(1, LOG_INTERVAL // world_size)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(RENDER_DIR, exist_ok=True)
    barrier()

    log_main(rank, "=" * 70)
    log_main(rank, "  VGGT GaussianHead training [nogt]: no GT + no data preprocessing")
    log_main(rank, "  Supervision = geo-init anchor (scale->spacing / opa->const) + render loss + opacity trio")
    log_main(rank, "  No GT param loss / no gt_gaussians / no gt_opa_std filtering")
    log_main(rank, "  Training images loaded per step from raw images (no train cache)")
    log_main(rank, f"  Views: random.sample({TRAIN_VIEWS_PER_STEP}); phase only anneals geo strength P1=1.0/P2=0.3/P3=0.05")
    log_main(rank, f"  geo: scale_k={GEO_SCALE_K} opa={GEO_OPA_CONST} w_scale={W_GEO_SCALE} w_opa={W_GEO_OPA}")
    log_main(rank, f"  Loss: RENDER={W_RENDER} BCE={W_OPA_BCE} ENT={W_OPA_ENTROPY} BAL={W_OPA_BALANCE}")
    log_main(rank, f"  HARD_STOP_STEP = {HARD_STOP_STEP} (compute budget)")
    log_main(rank, f"  Output: {OUTPUT_DIR} | world_size={world_size} | dtype={dtype}")
    log_main(rank, "=" * 70)

    # ---- 1. model ----
    log_main(rank, "\n[1/4] Loading VGGT-1B...")
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B", enable_gaussian=True).to(device)
    if hasattr(model.gaussian_head, 'frames_chunk_size'):
        model.gaussian_head.frames_chunk_size = GH_FRAMES_CHUNK_SIZE
        model.gaussian_head.use_checkpoint = GH_USE_CHECKPOINT
    if ENABLE_COLOR_HEAD:
        if hasattr(model.gaussian_head, 'enable_color_head_after_init'):
            model.gaussian_head.enable_color_head_after_init(device=device)
            log_main(rank, "  ColorHead enabled")
        elif getattr(model.gaussian_head, 'color_head', None) is not None:
            log_main(rank, "  ColorHead enabled by default")
    n_copied = copy_point_head_to_feature_head(model)
    log_main(rank, f"  Copied {n_copied} tensors point_head->dpt_feature_head [unfrozen]")
    for n, p in model.named_parameters():
        if 'gaussian_head' in n:
            p.requires_grad_(True)
        elif 'dpt_feature_head' in n:
            p.requires_grad_(not FREEZE_DPT_FEATURE_HEAD)
        else:
            p.requires_grad_(False)
    log_main(rank, f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if rank == 0:
        get_lpips_fn(device)
    log_main(rank, f"  elapsed: {format_duration(time.time()-t0)}")

    # ---- 2. scene list (no GT filtering, all used) ----
    log_main(rank, "\n[2/4] Scanning training scenes (no filtering, no cache)...")
    scenes = find_scenes(TRAIN_DIR, min_views=SCENE_MIN_VIEWS, max_views=SCENE_MAX_VIEWS, num_views=SCENE_NUM_VIEWS)
    assert len(scenes) > 0, f"no training scenes found under TRAIN_DIR='{TRAIN_DIR}' (set TRAIN_DIR)"
    log_main(rank, f"  Training scenes: {len(scenes)} | avg {sum(len(s) for s in scenes)/len(scenes):.1f} views/scene "
                   f"(all used; bad scenes handled by render robustness + skip)")

    # ---- 3. test-set precompute (lightweight) ----
    log_main(rank, "\n[3/4] Test-set precompute (lightweight, no GT)...")
    test_scenes = find_scenes(TEST_DIR, min_views=TEST_MIN_VIEWS, max_views=TEST_MAX_VIEWS, num_views=TEST_NUM_VIEWS)
    assert len(test_scenes) > 0, f"no test scenes found under TEST_DIR='{TEST_DIR}' (set TEST_DIR)"
    my_test = test_scenes[rank::world_size]
    my_test_idx = list(range(rank, len(test_scenes), world_size))
    barrier()
    precompute_test_split(model, my_test, device, dtype, TEST_CACHE_DIR, rank, my_test_idx)
    barrier()

    eval_test_data, all_test_data = None, None
    if rank == 0:
        all_test_data = []
        for idx in range(len(test_scenes)):
            name = Path(test_scenes[idx][0]).parent.name
            cp = os.path.join(TEST_CACHE_DIR, f"test_{idx:05d}_{name}.pt")
            if os.path.exists(cp):
                try:
                    all_test_data.append(torch.load(cp, map_location='cpu'))
                except Exception:
                    pass
        if len(all_test_data) > N_TEST_EVAL:
            random.Random(42).shuffle(all_test_data)
            eval_test_data = sorted(all_test_data[:N_TEST_EVAL], key=lambda d: d.get('scene_name', ''))
        else:
            eval_test_data = all_test_data
        log_main(rank, f"  test cache: {len(all_test_data)} | eval subset {len(eval_test_data)}")
    barrier()

    # ---- 4. init optimizer + xyz_offset + sanity ----
    log_main(rank, "\n[4/4] Init + xyz_offset / sanity (from scratch)...")
    if world_size > 1:
        for p in model.gaussian_head.parameters():
            dist.broadcast(p.data, src=0)
        if not FREEZE_DPT_FEATURE_HEAD:
            for p in model.dpt_feature_head.parameters():
                dist.broadcast(p.data, src=0)

    gh = model.gaussian_head
    param_groups = build_param_groups(gh, LR, xyz_lr_mult=XYZ_LR_MULTIPLIER, color_lr_mult=COLOR_LR_MULTIPLIER)
    if not FREEZE_DPT_FEATURE_HEAD:
        fh_params = [p for p in model.dpt_feature_head.parameters() if p.requires_grad]
        if fh_params:
            param_groups.append({'params': fh_params, 'lr': LR * FEATURE_HEAD_LR_MULT, 'name': 'feat'})
            log_main(rank, f"  dpt_feature_head added to optimizer: {sum(p.numel() for p in fh_params):,} params, lr={LR*FEATURE_HEAD_LR_MULT:.1e}")
    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    # ---- resume: resolve the checkpoint to restore ----
    resume_path = None
    if RESUME_CHECKPOINT:
        if RESUME_CHECKPOINT.lower() == 'auto':
            resume_path = find_resume_checkpoint(OUTPUT_DIR, EXP_TAG)
            if resume_path is None:
                log_main(rank, "  RESUME=auto but no resumable ckpt under OUTPUT_DIR; training from scratch")
        elif os.path.exists(RESUME_CHECKPOINT):
            resume_path = RESUME_CHECKPOINT
        else:
            log_main(rank, f"  RESUME='{RESUME_CHECKPOINT}' file not found; training from scratch")

    resume_gstep = 0
    resume_best_psnr, resume_best_ssim = 0.0, 0.0
    resume_best_lpips, resume_best_loss = float('inf'), float('inf')
    if resume_path:
        log_main(rank, f"\n  Resume: restoring from {resume_path}")
        (resume_gstep, resume_best_psnr, resume_best_ssim,
         resume_best_lpips, resume_best_loss) = load_checkpoint_for_resume(
            model, optimizer, resume_path, OUTPUT_DIR, EXP_TAG, device, rank)
        log_main(rank, f"      -> continue from global_step={resume_gstep} | "
                       f"best_psnr={resume_best_psnr:.3f} best_loss={resume_best_loss:.4f}")
        log_main(rank, f"      -> only PSNR>{resume_best_psnr:.3f} / loss<{resume_best_loss:.4f} will overwrite the corresponding best ckpt")

    # ---- xyz_offset: force-override only when training from scratch; keep the
    #      trained value from the ckpt on resume ----
    if not resume_path:
        old_val = gh.xyz_offset_log_scale.item()
        with torch.no_grad():
            gh.xyz_offset_log_scale.copy_(torch.tensor(XYZ_OFFSET_LOG_SCALE_OVERRIDE, device=device,
                                                       dtype=gh.xyz_offset_log_scale.dtype))
        log_main(rank, f"  xyz_offset_log_scale: {old_val:.4f} -> {gh.xyz_offset_log_scale.item():.4f} (exp={math.exp(gh.xyz_offset_log_scale.item()):.3f})")
    else:
        log_main(rank, f"  xyz_offset_log_scale (kept on resume): {gh.xyz_offset_log_scale.item():.4f} (exp={gh.xyz_offset_log_scale.exp().item():.3f})")

    if world_size > 1:
        for p in model.gaussian_head.parameters():
            dist.broadcast(p.data, src=0)
        if not FREEZE_DPT_FEATURE_HEAD:
            for p in model.dpt_feature_head.parameters():
                dist.broadcast(p.data, src=0)

    sanity_check_xyz_base(model, scenes[0], device, dtype, rank)
    barrier()

    # ---- scheduler: cosine anneal to HARD_STOP; fast-forward to resume_gstep on resume ----
    sched_total = HARD_STOP_LOCAL
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=WARMUP_FACTOR, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, sched_total - WARMUP_STEPS), eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_STEPS])
    if resume_gstep > 0:
        for _ in range(min(resume_gstep, sched_total)):
            scheduler.step()
        log_main(rank, f"  LR scheduler fast-forwarded to step {resume_gstep} (lr={optimizer.param_groups[0]['lr']:.2e})")
    log_main(rank, f"  LR anneal horizon (local steps): {sched_total:,}")

    # ---- training loop ----
    log_main(rank, "\n[Training loop] (pure geo_init, no GT, on-demand loading)...")
    log_main(rank, "-" * 70)
    current_dedup = DEDUP_METHOD
    n_mask_slow = 0; MASK_SLOW_THRESHOLD = 5
    # best metrics / global_step start from resume values (defaults when from scratch)
    best_loss = resume_best_loss; best_psnr = resume_best_psnr
    best_ssim = resume_best_ssim; best_lpips = resume_best_lpips
    train_start = time.time(); step_times = []
    no_improve = 0; early_stopped = False; n_local_skip = 0
    global_step = resume_gstep
    # resume positioning: skip already-trained epochs / scenes within the epoch
    # (the per-epoch shuffle is seeded by epoch, so it is reproducible)
    steps_per_epoch = max(1, len(scenes) // world_size)
    start_epoch = resume_gstep // steps_per_epoch
    skip_in_epoch = resume_gstep % steps_per_epoch
    if resume_gstep > 0:
        log_main(rank, f"  resume: continue from scene {skip_in_epoch} of epoch {start_epoch+1}/{NUM_EPOCHS}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if early_stopped:
            break
        g = random.Random(epoch * 1000 + 42)
        order = list(range(len(scenes))); g.shuffle(order)
        my_order = order[rank::world_size]
        start_idx = skip_in_epoch if epoch == start_epoch else 0

        for data_idx in my_order[start_idx:]:
            if early_stopped:
                break
            if global_step >= HARD_STOP_LOCAL:
                log_main(rank, f"\n  Hard stop (step {global_step} >= {HARD_STOP_LOCAL})")
                early_stopped = True; break

            random.seed(global_step * 7919 + 31 + rank)
            step_t0 = time.time()
            local_valid = True
            loss = None; avg_r_value = 0.0; n_render_ok = 0; S_sel = 0; H = W = 0; S_total = 0
            bce_val = ent_val = bal_val = 0.0
            mask_ratio = frame_std = 0.0; surf_val = mid_val = fog_val = 0.0
            opa_pf = [0.0, 0.0, 0.0, 0.0]; mask_el = 0.0; cd_val = 0.0
            geo_scl_val = geo_opa_val = 0.0; selected = []

            try:
                scene_imgs = scenes[data_idx]
                S_total = len(scene_imgs)
                images_input, gt_images_sel, selected, H, W = load_selected_views(
                    scene_imgs, TRAIN_VIEWS_PER_STEP, device)
                S_sel = len(selected)

                model.train()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images_input)

                if 'color_delta_norm' in predictions.get('gaussians', {}):
                    cd_raw = predictions['gaussians']['color_delta_norm']
                    cd_val = float(cd_raw) if not torch.is_tensor(cd_raw) else float(cd_raw.item())

                with torch.no_grad():
                    extr_pred, intr_pred = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=(H, W))
                    extr_pred = extr_pred.float(); intr_pred = intr_pred.float()

                # phase-anneal geo strength + render weight
                if global_step < PHASE2_LOCAL:
                    geo_scale, w_render = 1.0, W_RENDER
                elif global_step < PHASE3_LOCAL:
                    geo_scale, w_render = 0.3, W_RENDER * 2.0
                else:
                    geo_scale, w_render = 0.05, W_RENDER * 2.0

                with torch.no_grad():
                    xyz_base_pred = unproject_depth_to_world_torch(predictions['depth'].float(), extr_pred, intr_pred)
                    depth_conf_pred = predictions['depth_conf'].float()

                # Bad-scene guard: forward may produce NaN/Inf in degenerate
                # scenes. If those flow into the dedup mask / BCE / render they
                # trigger an unrecoverable device-side assert. Detect and raise
                # here before the CUDA error, so the existing skip fallback below
                # handles it (without corrupting the CUDA context). One .item()
                # sync, negligible cost.
                with torch.no_grad():
                    _g = predictions['gaussians']
                    _ok = (torch.isfinite(xyz_base_pred).all()
                           & torch.isfinite(depth_conf_pred).all()
                           & torch.isfinite(_g['xyz']).all()
                           & torch.isfinite(_g['scale']).all()
                           & torch.isfinite(_g['opacity']).all())
                if not bool(_ok.item()):
                    raise RuntimeError("nonfinite_pred")

                # geo-init cold-start anchor
                l_geo = torch.tensor(0.0, device=device)
                if geo_scale > 0:
                    l_geo, geo_scl_val, geo_opa_val = geo_init_param_loss(
                        predictions['gaussians'], xyz_base_pred, S_sel, H, W,
                        GEO_SCALE_K, GEO_OPA_CONST, W_GEO_SCALE, W_GEO_OPA, geo_scale)

                log_timing = (global_step < DEDUP_TIMING_LOG_EARLY or (global_step + 1) % DEDUP_TIMING_LOG_INTERVAL == 0)
                dedup_mask, mask_el, _ = safe_compute_dedup_mask(
                    method=current_dedup, xyz_base=xyz_base_pred, depth_conf=depth_conf_pred,
                    extr=extr_pred, intr=intr_pred, H=H, W=W, global_step=global_step,
                    log_timing=log_timing, rank=rank)
                if mask_el > DEDUP_TIMEOUT_SEC:
                    n_mask_slow += 1
                    if n_mask_slow >= MASK_SLOW_THRESHOLD and current_dedup != 'none':
                        log_all(rank, f"  mask consistently slow, switch to 'none'"); current_dedup = 'none'

                pred_opa = predictions['gaussians']['opacity'].float()
                l_bce = opacity_bce_loss(pred_opa, dedup_mask)
                l_ent = opacity_entropy_loss(pred_opa)
                l_bal = opacity_balance_loss(pred_opa)
                bce_val, ent_val, bal_val = float(l_bce.item()), float(l_ent.item()), float(l_bal.item())

                with torch.no_grad():
                    mask_ratio = float(dedup_mask.mean().item())
                    pf = dedup_mask[0].mean(dim=(1, 2)).cpu().numpy(); frame_std = float(pf.std())
                    surf_val = float((pred_opa > 0.5).float().mean().item())
                    mid_val = float(((pred_opa > 0.1) & (pred_opa < 0.5)).float().mean().item())
                    fog_val = float((pred_opa < 0.1).float().mean().item())
                    opa_pf = pred_opa[0].mean(dim=(1, 2)).cpu().numpy().tolist()[:4]
                    while len(opa_pf) < 4:
                        opa_pf.append(0.0)

                pred_scene = {k: v[0, :S_sel].reshape(-1, v.shape[-1]).float()
                              for k, v in predictions['gaussians'].items()
                              if isinstance(v, torch.Tensor) and v.dim() >= 4}

                r_losses = []; oom_fb = False
                for si in range(S_sel):
                    try:
                        rl, _ = compute_render_loss(pred_scene, gt_images_sel[si].float(),
                                                    extr_pred[0, si], intr_pred[0, si], H, W)
                        r_losses.append(rl)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            torch.cuda.empty_cache(); oom_fb = True; break
                        raise
                if oom_fb:
                    r_losses = []
                    ti = random.randint(0, S_sel - 1)
                    try:
                        rl, _ = compute_render_loss(pred_scene, gt_images_sel[ti].float(),
                                                    extr_pred[0, ti], intr_pred[0, ti], H, W)
                        r_losses.append(rl)
                    except RuntimeError:
                        torch.cuda.empty_cache()

                n_render_ok = len(r_losses)
                avg_r = (sum(r_losses) / len(r_losses)) if r_losses else torch.tensor(0.0, device=device)
                avg_r_value = float(avg_r.item())
                del pred_scene
                if n_render_ok == 0:
                    log_all(rank, f"  render all failed step{global_step} S={S_sel} H={H} W={W}")
                    raise RuntimeError("render_all_failed")

                loss = (l_geo + w_render * avg_r
                        + W_OPA_BCE * l_bce + W_OPA_ENTROPY * l_ent + W_OPA_BALANCE * l_bal)
                loss.backward()

            except RuntimeError as e:
                local_valid = False
                es = str(e).lower()
                bad_scene = scenes[data_idx][0] if 0 <= data_idx < len(scenes) else '?'
                bad_name = Path(bad_scene).parent.name if bad_scene != '?' else '?'
                if 'device-side assert' in es or 'cuda error' in es or 'illegal memory access' in es:
                    # Unrecoverable: the CUDA context is corrupted; even
                    # empty_cache re-raises the same error. Print the offending
                    # scene and exit cleanly, without any further CUDA calls.
                    log_all(rank, "\n" + "=" * 70)
                    log_all(rank, f"  Unrecoverable CUDA error @ step{global_step} — exiting")
                    log_all(rank, f"      offending scene: {bad_scene}")
                    log_all(rank, f"      scene dir: {Path(bad_scene).parent}  (S_total={S_total} H={H} W={W})")
                    log_all(rank, f"      original error: {e}")
                    log_all(rank, "      Suggestions: (1) move this scene dir out of TRAIN_DIR and retrain; or")
                    log_all(rank, "                   (2) add a filter to exclude such bad scenes; or")
                    log_all(rank, "                   (3) rerun with CUDA_LAUNCH_BLOCKING=1 to locate the offending kernel.")
                    log_all(rank, "=" * 70)
                    sys.stdout.flush()
                    os._exit(1)          # hard exit; do not touch the corrupted CUDA/NCCL state
                elif 'out of memory' in es:
                    log_all(rank, f"  OOM step{global_step} S={S_sel} H={H} W={W} scene={bad_name}, skip")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                elif 'nonfinite_pred' in es:
                    log_all(rank, f"  step{global_step} skipped (forward produced NaN/Inf) scene={bad_name}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                elif 'render_all_failed' in es:
                    log_all(rank, f"  step{global_step} skipped (render all failed) scene={bad_name}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                else:
                    log_all(rank, f"  step{global_step} failed scene={bad_name}: {e}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()

            n_valid = sync_gradients_flat_masked(gh, world_size, local_valid, device)
            if not FREEZE_DPT_FEATURE_HEAD:
                sync_gradients_flat_masked(model.dpt_feature_head, world_size, local_valid, device)
            if n_valid > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step(); scheduler.step()
            step_times.append(time.time() - step_t0)

            if rank == 0 and (global_step + 1) % LOG_LOCAL == 0:
                elapsed = time.time() - train_start
                avg_step = sum(step_times[-LOG_LOCAL:]) / len(step_times[-LOG_LOCAL:])
                remaining = avg_step * (HARD_STOP_LOCAL - global_step - 1)
                cur_lr = optimizer.param_groups[0]['lr']
                phase = "P1" if global_step < PHASE2_LOCAL else ("P2" if global_step < PHASE3_LOCAL else "P3")
                in_wm = global_step < WARMUP_STEPS
                wm = f"WM{global_step}/{WARMUP_STEPS} " if in_wm else ""
                xyz_off = gh.xyz_offset_log_scale.exp().item()
                mem = f" peak={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB" if torch.cuda.is_available() else ""
                if local_valid:
                    pf_str = ','.join(f'{x:.2f}' for x in opa_pf[:S_sel])
                    sel_str = ','.join(str(x) for x in selected[:5])
                    log_main(rank,
                             f"  [nogt|{phase}|{wm}] step{global_step:7d} ep{epoch+1}/{NUM_EPOCHS} | L={loss.item():.3f} "
                             f"rnd={avg_r_value:.3f}({n_render_ok}/{S_sel}v) geo=s{geo_scl_val:.2f}/o{geo_opa_val:.2f} "
                             f"bce={bce_val:.3f} ent={ent_val:.3f} bal={bal_val:.4f}\n"
                             f"               mask={mask_ratio:.2f}({current_dedup},{mask_el:.2f}s) fstd={frame_std:.3f} "
                             f"surf={surf_val:.3f} mid={mid_val:.3f} fog={fog_val:.3f} opa_pf=[{pf_str}] "
                             f"xyz_off={xyz_off:.3f} cd={cd_val:.4f} sel=[{sel_str}] "
                             f"valid={n_valid}/{world_size} lskip={n_local_skip} S={S_sel}/{S_total} H={H} W={W} "
                             f"lr={cur_lr:.1e}{mem} | {format_duration(elapsed)}/{format_duration(remaining)}")
                else:
                    log_main(rank, f"  [nogt|{phase}|{wm}] step{global_step:7d} rank0 SKIPPED "
                                   f"valid={n_valid}/{world_size} lskip={n_local_skip} | {format_duration(elapsed)}/{format_duration(remaining)}")

            if local_valid and loss is not None:
                cur_loss = loss.item()
                is_best_loss = cur_loss < best_loss
                if is_best_loss:
                    best_loss = cur_loss
            else:
                is_best_loss = False; cur_loss = best_loss

            is_eval = (global_step + 1) % EVAL_LOCAL == 0
            is_save = (global_step + 1) % SAVE_LOCAL == 0

            if is_eval:
                if rank == 0 and eval_test_data is not None:
                    p, s, lp, summary = evaluate_on_test_set(model, eval_test_data, device, dtype, global_step, RENDER_DIR, save_renders=3)
                    is_best = math.isfinite(p) and p < PSNR_SANE_MAX and p > best_psnr
                    if is_best:
                        best_psnr, best_ssim = p, s
                        if lp >= 0:
                            best_lpips = lp
                        no_improve = 0
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_test_{EXP_TAG}.pth"),
                                        metrics={'psnr': p, 'ssim': s, 'lpips': lp},
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    else:
                        no_improve += 1
                    lps = f" LPIPS={lp:.4f}" if lp >= 0 else ""
                    blps = f" LPIPS={best_lpips:.4f}" if best_lpips < float('inf') else ""
                    log_main(rank, f"    Eval ({len(eval_test_data)} scenes): PSNR={p:.2f}dB SSIM={s:.4f}{lps} "
                                   f"(best: {best_psnr:.2f}dB{blps}) [no_imp={no_improve}/{TEST_PATIENCE}] {summary}")
                    if is_best_loss:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_loss_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    if is_save:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_step{global_step+1}_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                stop = torch.zeros(1, device=device, dtype=torch.int)
                if rank == 0 and global_step >= MIN_TRAIN_LOCAL and no_improve >= TEST_PATIENCE:
                    stop[0] = 1
                if dist.is_initialized():
                    dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    early_stopped = True
                    log_main(rank, f"\n  Early stop (local step {global_step})"); break
                barrier(); model.train()
            elif is_save or is_best_loss:
                if rank == 0:
                    if is_best_loss:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_loss_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    if is_save:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_step{global_step+1}_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                if is_save:
                    barrier()

            global_step += 1
            # Explicitly free all large tensors of this step before clearing the
            # cache. empty_cache only returns unreferenced fragments; tensors still
            # bound to locals are kept. Delete each by literal name (a NameError on
            # an undefined name is ignored; note that deleting via locals()/exec
            # inside a function scope does not work — literal del is required).
            try: del predictions
            except Exception: pass
            try: del pred_scene
            except Exception: pass
            try: del xyz_base_pred
            except Exception: pass
            try: del depth_conf_pred
            except Exception: pass
            try: del dedup_mask
            except Exception: pass
            try: del pred_opa
            except Exception: pass
            try: del images_input
            except Exception: pass
            try: del gt_images_sel
            except Exception: pass
            try: del loss
            except Exception: pass
            try: del avg_r
            except Exception: pass
            try: del l_geo
            except Exception: pass
            try: del _g
            except Exception: pass
            try: del _ok
            except Exception: pass
            torch.cuda.empty_cache()

        if not early_stopped:
            barrier()
            log_main(rank, f"\n  === Epoch {epoch+1}/{NUM_EPOCHS} done | local step {global_step} | {format_duration(time.time()-train_start)} ===\n")

    # ---- final evaluation ----
    barrier()
    if rank == 0 and all_test_data is not None:
        log_main(rank, f"\n  Final full evaluation: {len(all_test_data)} test scenes...")
        fp, fs, flp, _ = evaluate_on_test_set(model, all_test_data, device, dtype, global_step, RENDER_DIR,
                                              save_renders=min(len(all_test_data), 20))
        log_main(rank, "\n" + "=" * 70)
        log_main(rank, "  Training complete! (no GT + no data preprocessing)")
        log_main(rank, "=" * 70)
        log_main(rank, f"  ---------- Final (all {len(all_test_data)} scenes) ----------")
        log_main(rank, f"  PSNR : {fp:.4f} dB | SSIM : {fs:.6f} | LPIPS: {flp:.6f}")
        log_main(rank, f"  ---------- Best during training ({len(eval_test_data)} scenes) ----------")
        log_main(rank, f"  PSNR : {best_psnr:.4f} dB | SSIM : {best_ssim:.6f}"
                       + (f" | LPIPS: {best_lpips:.6f}" if best_lpips < float('inf') else ""))
        log_main(rank, f"  Training time : {format_duration(time.time()-train_start)}")
        log_main(rank, f"  Best ckpt: {os.path.join(OUTPUT_DIR, f'gaussian_head_best_test_{EXP_TAG}.pth')}")
        log_main(rank, "=" * 70)

    barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()