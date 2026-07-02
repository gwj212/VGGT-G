# Usage:
#   CKPT_PATH=... TEST_CACHE_DIR=... OUTPUT_DIR=... python eval_only_5v.py
#
# Environment variables:
#   CKPT_PATH        path to the trained checkpoint (.pth)          [REQUIRED]
#   TEST_CACHE_DIR   dir with test_*.pt caches (from train_nogt.py) [REQUIRED]
#   OUTPUT_DIR       output dir (renders_final/ is written under it)[REQUIRED]
#   SAVE_VIEWS       comparison views saved per scene, 0=none. default 4 (head/mid/tail)
#   SKIP_LARGE_HW    if >0, skip scenes with H*W >= this threshold (avoid OOM). default: don't skip
#   DIAG_PER_FRAME   per-frame self-render diagnostic: 1=on (default), 0=off
#   VGGT_ROOT        repo root (defaults to this file's directory)

import os
import sys
import glob
import time
import math
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

VGGT_ROOT = os.environ.get("VGGT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.gs_optimizer import render_gaussians


# ============================================================
# Config  (set via env vars; defaults are intentionally blank)
# ============================================================

CKPT_PATH        = os.environ.get("CKPT_PATH", "")        # TODO: trained checkpoint (.pth)
TEST_CACHE_DIR   = os.environ.get("TEST_CACHE_DIR", "")   # TODO: dir with test_*.pt caches
OUTPUT_DIR       = os.environ.get("OUTPUT_DIR", "")       # TODO: output dir

RENDER_DIR       = os.path.join(OUTPUT_DIR, "renders_final")
SUMMARY_PATH     = os.path.join(OUTPUT_DIR, "eval_final_5v.txt")

SAVE_VIEWS       = int(os.environ.get("SAVE_VIEWS", "4"))
SKIP_LARGE_HW    = int(os.environ.get("SKIP_LARGE_HW", "0"))  # 0 = don't skip
DIAG_PER_FRAME   = int(os.environ.get("DIAG_PER_FRAME", "1")) != 0

GH_FRAMES_CHUNK_SIZE = 1
GH_USE_CHECKPOINT    = True

PSNR_CAP_DB = 100.0   # cap to avoid inf polluting the mean


# ============================================================
# Utilities
# ============================================================

def copy_point_head_to_feature_head(model):
    ph_state = model.point_head.state_dict()
    fh_state = model.dpt_feature_head.state_dict()
    copied = 0
    for name, param in ph_state.items():
        if name in fh_state and param.shape == fh_state[name].shape:
            fh_state[name].copy_(param)
            copied += 1
    model.dpt_feature_head.load_state_dict(fh_state)
    return copied


def compute_psnr(pred, gt):
    """Capped at PSNR_CAP_DB to avoid inf polluting the mean."""
    mse = F.mse_loss(pred.float(), gt.float()).item()
    if not math.isfinite(mse) or mse < 1e-10:
        return PSNR_CAP_DB
    val = 10.0 * math.log10(1.0 / mse)
    if not math.isfinite(val):
        return PSNR_CAP_DB
    return min(PSNR_CAP_DB, val)


def safe_mean(xs):
    """Mean over finite values only; empty -> 0.0."""
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def safe_min_max_med(xs):
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return 0.0, 0.0, 0.0
    s = sorted(finite)
    return s[0], s[-1], s[len(s) // 2]


def compute_ssim_val(pred, gt, window_size=11):
    p = pred.float().unsqueeze(0)
    g = gt.float().unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = p.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=p.dtype, device=p.device) - window_size // 2
    k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1d /= k1d.sum()
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
        p = pred.unsqueeze(0).float().to(device) * 2.0 - 1.0
        g = gt.unsqueeze(0).float().to(device) * 2.0 - 1.0
        return fn(p, g).item()


def save_render_comparison(rendered, gt, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)

    def to_pil(t):
        arr = (t.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr.transpose(1, 2, 0))

    r = to_pil(rendered)
    g_img = to_pil(gt)
    W_img, H_img = r.width, r.height
    canvas = Image.new("RGB", (W_img * 2 + 4, H_img), (128, 128, 128))
    canvas.paste(r, (0, 0))
    canvas.paste(g_img, (W_img + 4, 0))
    canvas.save(os.path.join(save_dir, fname))


# ============================================================
# Single-scene evaluation
# ============================================================

def evaluate_single_test_scene(model, test_data, device, dtype,
                                viz_view_indices=None, diag_per_frame=False):
    """
    Returns:
        avg_psnr (merged), avg_ssim, avg_lpips, viz_pack,
        n_ok, n_view,
        per_frame_diag : list of (view_idx, psnr_self, psnr_merged) — only when diag_per_frame=True
    """
    images_input = test_data['images_input'].to(device)
    S = test_data['num_views']
    H, W = test_data['H'], test_data['W']

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pred_eval = model(images_input)

        # merged Gaussians
        pred_scene = {
            k: v[0].reshape(-1, v.shape[-1]).float()
            for k, v in pred_eval['gaussians'].items()
            if torch.is_tensor(v) and v.ndim >= 2
        }

        psnrs, ssims, lpipss = [], [], []
        per_frame_diag = []
        viz_pack = {}
        viz_set = set(viz_view_indices) if viz_view_indices is not None else set()
        use_merged = True       # after a merged-render OOM, fall back to per-frame
        diag_disabled_oom = False  # after a diagnostic-render OOM, stop diagnostics

        for s in range(S):
            ext = test_data['exts'][s].to(device)
            itr = test_data['itrs'][s].to(device)
            gt  = test_data['gt_images'][s].to(device)

            # ============================================================
            # (1) merged render — the main metric
            # ============================================================
            rendered = None
            try:
                if use_merged:
                    rendered = render_gaussians(
                        pred_scene['xyz'], pred_scene['rotation'],
                        pred_scene['scale'], pred_scene['opacity'],
                        pred_scene['color'], ext, itr, H, W,
                    )
                else:
                    ef = {k: v[0, s].float() for k, v in pred_eval['gaussians'].items()}
                    rendered = render_gaussians(
                        ef['xyz'], ef['rotation'], ef['scale'],
                        ef['opacity'], ef['color'], ext, itr, H, W,
                    )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and use_merged:
                    torch.cuda.empty_cache()
                    use_merged = False
                    diag_disabled_oom = True   # memory tight; stop diagnostics too
                    try:
                        ef = {k: v[0, s].float()
                        for k, v in pred_eval['gaussians'].items()
                        if torch.is_tensor(v) and v.ndim >= 2
                        }
                        rendered = render_gaussians(
                            ef['xyz'], ef['rotation'], ef['scale'],
                            ef['opacity'], ef['color'], ext, itr, H, W,
                        )
                    except Exception:
                        torch.cuda.empty_cache()
                        del ext, itr, gt
                        continue
                else:
                    torch.cuda.empty_cache()
                    del ext, itr, gt
                    continue

            psnr_merged = compute_psnr(rendered, gt)
            ssim_s      = compute_ssim_val(rendered, gt)
            lp_s        = compute_lpips(rendered, gt, device)

            psnrs.append(psnr_merged)
            ssims.append(ssim_s)
            if lp_s >= 0:
                lpipss.append(lp_s)

            # ============================================================
            # (2) per-frame self-render diagnostic
            # ============================================================
            psnr_self = float('nan')
            if diag_per_frame and not diag_disabled_oom:
                try:
                    pred_self = {
                        k: v[0, s].reshape(-1, v.shape[-1]).float()
                        for k, v in pred_eval['gaussians'].items()
                        if torch.is_tensor(v) and v.ndim >= 2
                    }
                    rendered_self = render_gaussians(
                        pred_self['xyz'], pred_self['rotation'],
                        pred_self['scale'], pred_self['opacity'],
                        pred_self['color'], ext, itr, H, W,
                    )
                    psnr_self = compute_psnr(rendered_self, gt)
                    per_frame_diag.append((s, psnr_self, psnr_merged))
                    del pred_self, rendered_self
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        diag_disabled_oom = True
                    torch.cuda.empty_cache()
                except Exception:
                    torch.cuda.empty_cache()

            # ============================================================
            # (3) store comparison images
            # ============================================================
            if s in viz_set:
                viz_pack[s] = (
                    rendered.detach().cpu(),
                    gt.detach().cpu(),
                    psnr_merged, ssim_s, lp_s,
                    psnr_self,   # may be nan
                )

            del rendered, gt, ext, itr
            torch.cuda.empty_cache()

        del pred_scene, pred_eval, images_input

    avg_psnr  = safe_mean(psnrs)
    avg_ssim  = safe_mean(ssims)
    avg_lpips = safe_mean(lpipss) if lpipss else -1.0
    return (avg_psnr, avg_ssim, avg_lpips,
            viz_pack, len(psnrs), S, per_frame_diag)


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ============================================================
# Main
# ============================================================

def main():
    t_total = time.time()

    assert torch.cuda.is_available(), "GPU required"
    device = "cuda:0"
    cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    print("=" * 70)
    print("  Evaluation only — load ckpt, run all scenes (inf-bug fix + per-frame diagnostic)")
    print("=" * 70)
    print(f"  ckpt           : {CKPT_PATH}")
    print(f"  test cache dir : {TEST_CACHE_DIR}")
    print(f"  render out dir : {RENDER_DIR}")
    print(f"  summary path   : {SUMMARY_PATH}")
    print(f"  device         : {device}  cap={cap}  dtype={dtype}")
    print(f"  SAVE_VIEWS     : {SAVE_VIEWS} views/scene")
    print(f"  DIAG_PER_FRAME : {DIAG_PER_FRAME}  (per-frame self-render diagnostic)")
    if SKIP_LARGE_HW > 0:
        print(f"  SKIP_LARGE_HW  : skip scenes with H*W >= {SKIP_LARGE_HW}")
    print(f"  PSNR_CAP_DB    : {PSNR_CAP_DB}  (cap, avoid inf pollution)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)

    # ===== 1. load model =====
    print("\n[1/4] Loading VGGT-1B...")
    t = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B", enable_gaussian=True).to(device)
    if hasattr(model.gaussian_head, 'frames_chunk_size'):
        model.gaussian_head.frames_chunk_size = GH_FRAMES_CHUNK_SIZE
        model.gaussian_head.use_checkpoint = GH_USE_CHECKPOINT
    n_copied = copy_point_head_to_feature_head(model)
    print(f"  Copied {n_copied} dpt_feature_head tensors (fallback; overwritten by ckpt)")
    print(f"  elapsed: {format_duration(time.time() - t)}")

    # ===== 2. load ckpt =====
    print("\n[2/4] Loading ckpt...")
    assert os.path.isfile(CKPT_PATH), f"ckpt not found: {CKPT_PATH} (set CKPT_PATH)"
    t = time.time()
    ckpt = torch.load(CKPT_PATH, map_location=device)

    missing_gh, unexpected_gh = model.gaussian_head.load_state_dict(
        ckpt['gaussian_head_state_dict'], strict=False
    )
    print(f"  gaussian_head: missing={len(missing_gh)} unexpected={len(unexpected_gh)}")

    if 'dpt_feature_head_state_dict' in ckpt:
        missing_fh, unexpected_fh = model.dpt_feature_head.load_state_dict(
            ckpt['dpt_feature_head_state_dict'], strict=False
        )
        print(f"  dpt_feature_head: missing={len(missing_fh)} unexpected={len(unexpected_fh)}")
    else:
        print(f"  ckpt has no dpt_feature_head (frozen), keep weights copied from point_head")

    print(f"  ckpt step={ckpt.get('global_step', '?')} "
          f"epoch={ckpt.get('epoch', '?')} "
          f"loss={ckpt.get('loss', '?')}")
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        psnr_v  = m.get('psnr', float('nan'))
        ssim_v  = m.get('ssim', float('nan'))
        lpips_v = m.get('lpips', float('nan'))
        print(f"  metrics recorded at training time: PSNR={psnr_v:.4f} "
              f"SSIM={ssim_v:.6f} LPIPS={lpips_v:.6f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    get_lpips_fn(device)
    print(f"  elapsed: {format_duration(time.time() - t)}")

    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"  current GPU memory: free={free_b/1024**3:.1f}GB / total={total_b/1024**3:.1f}GB")

    # ===== 3. load all test caches =====
    print("\n[3/4] Loading all test caches...")
    t = time.time()
    cache_files = sorted(glob.glob(os.path.join(TEST_CACHE_DIR, "test_*.pt")))
    print(f"  found {len(cache_files)} test cache files")
    assert len(cache_files) > 0, f"no test cache found: {TEST_CACHE_DIR}/test_*.pt (set TEST_CACHE_DIR)"

    test_data_list = []
    n_skip_large = 0
    for cf in cache_files:
        try:
            td = torch.load(cf, map_location='cpu')
            assert all(k in td for k in
                       ['gt_images', 'images_input', 'exts', 'itrs',
                        'num_views', 'H', 'W'])
            if SKIP_LARGE_HW > 0 and td['H'] * td['W'] >= SKIP_LARGE_HW:
                n_skip_large += 1
                continue
            test_data_list.append(td)
        except Exception as e:
            print(f"  skip corrupted {Path(cf).name}: {e}")

    print(f"  usable scenes {len(test_data_list)} "
          f"(skipped corrupted {len(cache_files) - len(test_data_list) - n_skip_large}, "
          f"skipped too-large {n_skip_large}) "
          f"| {format_duration(time.time() - t)}")

    # ===== 4. evaluate =====
    print("\n[4/4] Evaluating...")
    t_eval = time.time()

    psnrs, ssims, lpipss = [], [], []
    # per-frame diagnostic accumulators
    f0_self_psnrs       = []   # frame 0 self-render PSNR
    f0_merged_psnrs     = []   # frame 0 merged-render PSNR
    nonf0_self_psnrs    = []   # frames 1..S-1 self-render PSNR
    nonf0_merged_psnrs  = []   # frames 1..S-1 merged-render PSNR
    diag_capped_count   = 0    # how many frames hit PSNR_CAP_DB (suspicious)

    n_failed = 0
    n_total = len(test_data_list)
    log_lines = []

    for i, td in enumerate(test_data_list):
        S_i = td['num_views']
        H_i, W_i = td['H'], td['W']
        scene_name = td.get('scene_name', f'scene{i:03d}')

        # pick SAVE_VIEWS views to save (head/mid/tail)
        if SAVE_VIEWS <= 0:
            viz_views = None
        elif SAVE_VIEWS >= 3:
            viz_views = sorted(set([0, S_i // 2, S_i - 1]))
        else:
            viz_views = list(range(min(SAVE_VIEWS, S_i)))

        try:
            (psnr_v, ssim_v, lpips_v, viz_pack,
             n_ok, n_view, per_frame_diag) = evaluate_single_test_scene(
                model, td, device, dtype, viz_views,
                diag_per_frame=DIAG_PER_FRAME,
            )
        except Exception as e:
            print(f"  [{i+1:3d}/{n_total}] {scene_name[:28]:28s} failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            n_failed += 1
            continue

        if n_ok == 0:
            print(f"  [{i+1:3d}/{n_total}] {scene_name[:28]:28s} all views failed to render "
                  f"H={H_i} W={W_i} S={n_view}")
            n_failed += 1
            continue

        psnrs.append(psnr_v)
        ssims.append(ssim_v)
        if lpips_v >= 0:
            lpipss.append(lpips_v)

        # accumulate per-frame diagnostic
        scene_f0_self = float('nan')
        scene_f0_merged = float('nan')
        for (view_idx, p_self, p_merged) in per_frame_diag:
            if p_self >= PSNR_CAP_DB - 0.01 or p_merged >= PSNR_CAP_DB - 0.01:
                diag_capped_count += 1
            if view_idx == 0:
                f0_self_psnrs.append(p_self)
                f0_merged_psnrs.append(p_merged)
                scene_f0_self = p_self
                scene_f0_merged = p_merged
            else:
                nonf0_self_psnrs.append(p_self)
                nonf0_merged_psnrs.append(p_merged)

        # save comparison images (viz_pack now carries an extra psnr_self)
        if viz_views is not None and viz_pack:
            for v_idx in viz_views:
                if v_idx not in viz_pack:
                    continue
                rendered, gt_img, p, s, lp, p_self = viz_pack[v_idx]
                lp_str = f"_lpips{lp:.4f}" if lp >= 0 else ""
                self_str = (f"_slf{p_self:.2f}"
                            if math.isfinite(p_self) else "")
                fname = (f"{i:03d}_{scene_name}_v{v_idx}_"
                         f"psnr{p:.2f}{self_str}_ssim{s:.4f}{lp_str}.png")
                save_render_comparison(rendered, gt_img, RENDER_DIR, fname)

        # single-line log
        diag_str = ""
        if DIAG_PER_FRAME and math.isfinite(scene_f0_self):
            diag_str = (f" | f0:slf={scene_f0_self:5.2f}/mrg={scene_f0_merged:5.2f}")
        log_line = (f"  [{i+1:3d}/{n_total}] {scene_name[:28]:28s} "
                    f"H={H_i:3d} W={W_i:3d} S={n_view:2d} ok={n_ok:2d} | "
                    f"PSNR={psnr_v:5.2f} SSIM={ssim_v:.4f} "
                    f"LPIPS={lpips_v if lpips_v >= 0 else float('nan'):.4f}"
                    f"{diag_str}")
        log_lines.append(log_line)

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            avg_p = safe_mean(psnrs)
            avg_s = safe_mean(ssims)
            avg_l = safe_mean(lpipss) if lpipss else -1.0
            elapsed = time.time() - t_eval
            eta = elapsed / (i + 1) * (n_total - i - 1)
            extra = ""
            if DIAG_PER_FRAME and f0_self_psnrs:
                f0s   = safe_mean(f0_self_psnrs)
                nf0s  = safe_mean(nonf0_self_psnrs) if nonf0_self_psnrs else 0.0
                extra = f" | f0_slf={f0s:5.2f} nf0_slf={nf0s:5.2f}"
            print(f"  [{i+1:3d}/{n_total}] avg PSNR={avg_p:.2f} "
                  f"SSIM={avg_s:.4f} LPIPS={avg_l:.4f}{extra} "
                  f"| failed {n_failed} | "
                  f"{format_duration(elapsed)} / left {format_duration(eta)}")

        # clean up between scenes
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            gc.collect()

    # ===== summary =====
    print("\n" + "=" * 70)
    print(f"  Evaluation complete!  total time {format_duration(time.time() - t_total)}")
    print("=" * 70)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append(f"  ckpt: {CKPT_PATH}")
    summary_lines.append(f"  ckpt step: {ckpt.get('global_step', '?')}")
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        summary_lines.append(
            f"  recorded at training time (subset): "
            f"PSNR={m.get('psnr', float('nan')):.4f} "
            f"SSIM={m.get('ssim', float('nan')):.6f} "
            f"LPIPS={m.get('lpips', float('nan')):.6f}"
        )
    summary_lines.append("-" * 70)
    summary_lines.append(f"  Full-set evaluation (this run):")
    summary_lines.append(f"  successful scenes: {len(psnrs)}/{n_total}  failed {n_failed}")
    if psnrs:
        avg_p = safe_mean(psnrs)
        avg_s = safe_mean(ssims)
        avg_l = safe_mean(lpipss) if lpipss else -1.0
        min_p, max_p, med_p = safe_min_max_med(psnrs)
        n_capped_scenes = sum(1 for x in psnrs if x >= PSNR_CAP_DB - 0.01)
        summary_lines.append(f"  PSNR  avg = {avg_p:.4f} dB   "
                             f"min = {min_p:.2f}   med = {med_p:.2f}   max = {max_p:.2f}")
        if n_capped_scenes > 0:
            summary_lines.append(
                f"  {n_capped_scenes} scenes hit the PSNR cap {PSNR_CAP_DB:.0f}dB; "
                f"possibly pure-black/solid-color GT, please double-check"
            )
        summary_lines.append(f"  SSIM  avg = {avg_s:.6f}")
        if lpipss:
            summary_lines.append(f"  LPIPS avg = {avg_l:.6f}   "
                                 f"(over {len(lpipss)} scenes)")

    # per-frame diagnostic summary
    if DIAG_PER_FRAME and (f0_self_psnrs or nonf0_self_psnrs):
        summary_lines.append("-" * 70)
        summary_lines.append(f"  Per-frame self-render diagnostic:")
        if f0_self_psnrs:
            f0_self_avg   = safe_mean(f0_self_psnrs)
            f0_merged_avg = safe_mean(f0_merged_psnrs)
            summary_lines.append(
                f"    frame 0    ({len(f0_self_psnrs):3d} frames):  "
                f"self = {f0_self_avg:5.2f} dB  |  merged = {f0_merged_avg:5.2f} dB  "
                f"|  d = {f0_self_avg - f0_merged_avg:+.2f} dB"
            )
        if nonf0_self_psnrs:
            nf_self_avg   = safe_mean(nonf0_self_psnrs)
            nf_merged_avg = safe_mean(nonf0_merged_psnrs)
            summary_lines.append(
                f"    non-frame-0 ({len(nonf0_self_psnrs):3d} frames):  "
                f"self = {nf_self_avg:5.2f} dB  |  merged = {nf_merged_avg:5.2f} dB  "
                f"|  d = {nf_self_avg - nf_merged_avg:+.2f} dB"
            )
        if f0_self_psnrs and nonf0_self_psnrs:
            f0_self_avg = safe_mean(f0_self_psnrs)
            nf_self_avg = safe_mean(nonf0_self_psnrs)
            gap = f0_self_avg - nf_self_avg
            summary_lines.append(
                f"    self-render gap (f0 - nf0) = {gap:+.2f} dB"
            )
        if diag_capped_count > 0:
            summary_lines.append(
                f"    {diag_capped_count} diagnostic frames hit the PSNR cap {PSNR_CAP_DB:.0f}dB"
            )

    summary_lines.append("-" * 70)
    summary_lines.append(f"  renders: {RENDER_DIR}")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("per-scene details:")
    summary_lines.extend(log_lines)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n  summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()