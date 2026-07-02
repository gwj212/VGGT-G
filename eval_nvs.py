#!/usr/bin/env python3
"""
eval_nvs.py — novel-view-synthesis (NVS) evaluation for the GaussianHead.
(Renamed from compare_fair_loo_v4.py; the third-party comparison was removed,
 so this now evaluates only this model's novel-view synthesis.)

Protocol (leak-free, standard NVS):
  1. The model sees only the context views; the test views are never fed in.
  2. Gaussians are rendered at the held-out GROUND-TRUTH poses (GT extrinsics),
     which removes the pose-estimation confounder.
  3. Alignment uses the exact GT-ctx <-> predicted-ctx correspondence: the
     rotation is averaged from the full GT camera orientations and a single
     global scale comes from pairwise camera-center distances, so near-collinear
     or clustered cameras do not degenerate.
  4. GT extrinsics / intrinsics / source-image names / original sizes all come
     from a per-scene meta.json; the source images are read directly from the
     json's src_dir (no dependency on renamed files, no COLMAP-bin parsing).

Metrics:
  - full-frame PSNR / SSIM / LPIPS over the whole test image.
  - coverage: the fraction of the test frame the model actually fills.
  - masked PSNR / SSIM inside the model's coverage mask (fidelity of the
    reconstructed region only).

Timing (CUDA-synchronized, does not affect any metric):
  - one-off model load (VGGT + GaussianHead), amortized over all scenes;
  - single forward pass model(...), grouped by the number of context views.

First run — please verify (the script prints these; flip the switch if wrong):
  - GT_EXTRINSIC_C2W: the extrinsic convention in meta.json. COLMAP writes
    world-to-camera, so the default is False. If the ctx self-check ([GT-SELF]
    below) is very low, or renders are clearly misaligned, flip it.

Usage:
  DATASET_DIR=... CKPT_PATH=... python eval_nvs.py
"""

import os, sys, glob, time, math, random, contextlib, traceback, json, struct

# ============================================================
#  Config  (set paths via env vars or edit the defaults below)
# ============================================================

DATASET_DIR  = os.environ.get("DATASET_DIR", "")   # TODO: root holding the per-scene meta.json files
META_GLOB    = "*.json"          # metadata-json pattern inside each scene dir
# If the original image tree moved: replace this prefix of the json's src_dir; None = no remap
SRC_DIR_REMAP = None             # e.g. ("/old/prefix", "/new/prefix")
GT_EXTRINSIC_C2W = False         # extrinsic convention in the json (COLMAP w2c => False)

CKPT_PATH    = os.environ.get("CKPT_PATH", "")     # TODO: trained checkpoint (.pth)

SAVE_DIR_IMGS   = os.environ.get("SAVE_DIR_IMGS", "")    # TODO: side-by-side (pred vs GT) images; blank = don't save
SAVE_DIR_PLY    = os.environ.get("SAVE_DIR_PLY", "")     # TODO: exported .ply; blank = don't export
LOG_FILE        = os.environ.get("LOG_FILE", "")         # TODO: summary log file; blank = don't write
SAVE_DIR_GH_IMG = os.environ.get("SAVE_DIR_GH_IMG", "")  # TODO: single prediction images (for figures); blank = don't save

EXPORT_GH_GAUSSIANS_PLY  = True
EXPORT_GH_POINTCLOUD_PLY = True
PLY_OPA_LOW              = 0.1
PLY_OPA_HIGH             = 0.5

XYZ_BASE_SOURCE = "depth_unproject"

N_SCENES     = 99
CTX_TEST_MAP = {3: 1, 6: 1, 10: 2}
CTX_COUNTS   = sorted(CTX_TEST_MAP.keys())
SEED         = 42

VGGT_ROOT     = os.environ.get("VGGT_ROOT", os.path.dirname(os.path.abspath(__file__)))
HF_HOME       = None             # optional: point to your own Hugging Face cache dir

# Reasonable range for the estimated GT->model scale (pairwise distance ratio);
# clamp + warn outside it.
ALIGN_SCALE_CLAMP = (0.02, 50.0)

# Intrinsics for rendering: the model renders at GT-anchored poses using its OWN
# predicted intrinsics. VGGT's intrinsics are ~GT here, so predicted ~ GT (the
# numbers barely change), but keeping "predicted" makes the protocol self-consistent.
GH_USE_GT_INTRINSIC = False      # False = use VGGT's predicted intrinsics (recommended);
                                 # True  = use GT intrinsics (VGGT estimates them accurately, ~equal)

EVAL_LONG_SIDE = 448

RENDER_BG_WHITE = True           # composite GH's black background onto white

EVAL_MASKED    = True
MASK_ALPHA_THR = 0.5             # alpha threshold that defines the coverage mask

DEBUG_SELF_CHECK = True          # render one ctx GT view as a self-check (validates alignment + Gaussians)
DEBUG_SELF_CHECK_LIMIT = 6
DEBUG_ALIGN_PRINT_LIMIT = 16

# ============================================================

if XYZ_BASE_SOURCE is not None:
    os.environ["VGGT_XYZ_BASE_SOURCE"] = XYZ_BASE_SOURCE
if HF_HOME is not None:
    os.environ["HF_HOME"] = HF_HOME

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.gs_optimizer import render_gaussians


# ===========================================================
#  Metrics
# ===========================================================

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return 10 * math.log10(1 / mse) if mse > 1e-10 else float("inf")


def compute_ssim(pred, gt, ws=11):
    p, g = pred.float().unsqueeze(0), gt.float().unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    ch = p.shape[1]; sigma = 1.5
    coords = torch.arange(ws, dtype=p.dtype, device=p.device) - ws // 2
    k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d /= k1d.sum()
    ker = k1d.outer(k1d).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = ws // 2
    mu1 = F.conv2d(p, ker, padding=pad, groups=ch)
    mu2 = F.conv2d(g, ker, padding=pad, groups=ch)
    m1sq, m2sq, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1  = F.conv2d(p*p, ker, padding=pad, groups=ch) - m1sq
    s2  = F.conv2d(g*g, ker, padding=pad, groups=ch) - m2sq
    s12 = F.conv2d(p*g, ker, padding=pad, groups=ch) - m12
    return (((2*m12+C1)*(2*s12+C2)) / ((m1sq+m2sq+C1)*(s1+s2+C2))).mean().item()


def compute_psnr_masked(pred, gt, mask, eps=1e-10):
    m = mask.float().to(pred.device); n = m.sum().item()
    if n < 10: return float("nan")
    se = ((pred.float() - gt.float()) ** 2) * m.unsqueeze(0)
    mse = se.sum().item() / (n * pred.shape[0])
    return 10 * math.log10(1 / mse) if mse > eps else float("inf")


def _ssim_map(pred, gt, ws=11):
    p, g = pred.float().unsqueeze(0), gt.float().unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    ch = p.shape[1]; sigma = 1.5
    coords = torch.arange(ws, dtype=p.dtype, device=p.device) - ws // 2
    k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d /= k1d.sum()
    ker = k1d.outer(k1d).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = ws // 2
    mu1 = F.conv2d(p, ker, padding=pad, groups=ch)
    mu2 = F.conv2d(g, ker, padding=pad, groups=ch)
    m1sq, m2sq, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1  = F.conv2d(p*p, ker, padding=pad, groups=ch) - m1sq
    s2  = F.conv2d(g*g, ker, padding=pad, groups=ch) - m2sq
    s12 = F.conv2d(p*g, ker, padding=pad, groups=ch) - m12
    smap = ((2*m12+C1)*(2*s12+C2)) / ((m1sq+m2sq+C1)*(s1+s2+C2))
    return smap.mean(1).squeeze(0)


def compute_ssim_masked(pred, gt, mask):
    smap = _ssim_map(pred, gt); m = mask.float().to(smap.device)
    if m.sum() < 10: return float("nan")
    return float((smap * m).sum() / m.sum())


_lpips_fn = None
def compute_lpips(pred, gt, device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            for p in _lpips_fn.parameters(): p.requires_grad_(False)
            print("  LPIPS(VGG) loaded")
        except ImportError:
            _lpips_fn = "unavailable"
    if _lpips_fn == "unavailable": return -1.0
    with torch.no_grad():
        p = pred.unsqueeze(0).float().to(device) * 2 - 1
        g = gt.unsqueeze(0).float().to(device) * 2 - 1
        return _lpips_fn(p, g).item()


def format_dur(s):
    h, m = int(s // 3600), int((s % 3600) // 60); sec = int(s % 60)
    return f"{h}h{m}m{sec}s" if h else (f"{m}m{sec}s" if m else f"{sec}s")


# ===========================================================
#  Forward / load timing
#    GPU kernels are async -> we must cuda.synchronize() before/after, otherwise
#    we would time kernel launches instead of real work. Model load is a one-off
#    cost (amortized over all scenes) and is reported separately.
# ===========================================================

_timing = {
    "gh_load":  None,   # VGGT + GaussianHead load (one-off, s)
    "gh_fwd":   {},     # {n_ctx: [s, ...]}  pure forward (network inference)
    "gh_pre":   {},     # {n_ctx: [s, ...]}  image read + preprocess (not a forward)
}


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _timed(store, key):
    """CUDA-synchronized timing; the elapsed time is appended to store[key] (a list)."""
    _cuda_sync(); _t0 = time.time()
    try:
        yield
    finally:
        _cuda_sync(); store.setdefault(key, []).append(time.time() - _t0)


def _flatten(d):
    out = []
    for k in d: out.extend(d[k])
    return out


def _fwd_stats(vals):
    """Return (median, fastest, mean, n). median/fastest are unaffected by the first CUDA warmup."""
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    sv = sorted(vals)
    return sv[len(sv)//2], sv[0], sum(vals)/len(vals), len(vals)


# ===========================================================
#  Common evaluation resolution
# ===========================================================

def canonical_hw(w, h, long_side=EVAL_LONG_SIDE):
    """Long side = long_side, both sides rounded to a multiple of 14, aspect kept. Returns (H, W)."""
    if w >= h:
        W = long_side; H = max(14, round(h * long_side / w / 14) * 14)
    else:
        H = long_side; W = max(14, round(w * long_side / h / 14) * 14)
    return H, W


def load_canonical_gt_with_size(path, long_side=EVAL_LONG_SIDE):
    """Return (gt(3,H,W) in [0,1], (W0,H0) original size, (H,W) eval size)."""
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    H, W = canonical_hw(W0, H0, long_side)
    img2 = img.resize((W, H), Image.BICUBIC)
    arr = np.asarray(img2).astype(np.float32) / 255.0
    gt = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return gt, (W0, H0), (H, W)


def resize_chw(img, hw):
    if img.shape[-2:] == tuple(hw):
        return img.float().clamp(0, 1)
    out = F.interpolate(img.unsqueeze(0).float(), size=tuple(hw),
                        mode="bilinear", align_corners=False, antialias=True)
    return out.squeeze(0).clamp(0, 1)


def composite_premult_on_white(rgb_on_black, alpha):
    return (rgb_on_black.float() + (1.0 - alpha.float())).clamp(0.0, 1.0)


def rescale_intrinsic_pixel(K, orig_wh, eval_hw):
    """K(3,3) in the original (W0,H0) pixel frame -> the eval-resolution (H_e,W_e) pixel frame."""
    W0, H0 = orig_wh; He, We = eval_hw
    sx = We / float(W0); sy = He / float(H0)
    K2 = K.clone().float()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2


# ===========================================================
#  Pose-convention tools + GT-anchored similarity transform
#    COLMAP -> model: center_pred ~ s*Ralign*center_gt + talign,
#                     Rc2w_pred  ~ Ralign*Rc2w_gt.
#    The rotation is fixed from the full GT orientations (degeneracy-free) and
#    the scale from pairwise camera-center distances.
# ===========================================================

def pose_to_center_Rc2w(E, is_c2w):
    """E:(...,>=3,4). Returns (center(3,), Rc2w(3,3))."""
    if is_c2w:
        Rc2w = E[:3, :3].float()
        center = E[:3, 3].float()
    else:  # w2c
        R = E[:3, :3].float(); t = E[:3, 3].float()
        Rc2w = R.transpose(-1, -2)
        center = -(Rc2w @ t)
    return center, Rc2w


def center_Rc2w_to_w2c(center, Rc2w):
    R = Rc2w.transpose(-1, -2)
    t = -(R @ center)
    E = torch.eye(4, device=center.device, dtype=torch.float32)
    E[:3, :3] = R; E[:3, 3] = t
    return E


def center_Rc2w_to_c2w(center, Rc2w):
    E = torch.eye(4, device=center.device, dtype=torch.float32)
    E[:3, :3] = Rc2w; E[:3, 3] = center
    return E


def rotation_average(Rs):
    """Rs:(N,3,3) -> the average rotation (SVD projection onto SO(3))."""
    M = Rs.double().sum(0)
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone(); U[:, -1] *= -1; R = U @ Vh
    return R.float()


def _pairdist_mean(c):
    if c.shape[0] < 2:
        return torch.tensor(1.0, device=c.device)
    d = torch.cdist(c, c)
    iu = torch.triu_indices(c.shape[0], c.shape[0], offset=1)
    return d[iu[0], iu[1]].mean()


def compute_gt_to_model_sim3(gt_ctx_E, pred_ctx_E, gt_is_c2w, pred_is_c2w):
    """Return (s, Ralign, talign) mapping the GT (COLMAP) frame into the model frame.
    Robust version: an occasional bad predicted pose (VGGT mis-estimating a view)
    can throw off the rotation/scale -> the mapped camera points backwards -> a
    fully white render. First estimate one rotation over all views, drop outlier
    frames by their per-frame rotation angle (> 3x median and > 15 deg), then
    re-estimate rotation/scale/translation from the inliers."""
    n = gt_ctx_E.shape[0]
    cg, cp, Rel = [], [], []
    for i in range(n):
        c1, R1 = pose_to_center_Rc2w(gt_ctx_E[i], gt_is_c2w)
        c2, R2 = pose_to_center_Rc2w(pred_ctx_E[i], pred_is_c2w)
        cg.append(c1); cp.append(c2)
        Rel.append(R2 @ R1.transpose(-1, -2))      # Rc2w_pred @ Rc2w_gt^T
    cg = torch.stack(cg); cp = torch.stack(cp); Rel = torch.stack(Rel)
    R0 = rotation_average(Rel)
    # geodesic angle of each frame wrt R0 = arccos((trace(Rel*R0^T)-1)/2)
    RR0 = torch.einsum('nij,kj->nik', Rel, R0)               # Rel @ R0^T
    cos = ((torch.einsum('nii->n', RR0) - 1.0) * 0.5).clamp(-1, 1)
    angs = torch.arccos(cos)
    thr = torch.maximum(angs.median() * 3.0,
                        torch.deg2rad(torch.tensor(15.0, device=angs.device)))
    keep = angs <= thr
    if int(keep.sum()) < 2:
        keep = torch.ones_like(keep)
    n_drop = int((~keep).sum())
    Ralign = rotation_average(Rel[keep])
    cgk, cpk = cg[keep], cp[keep]
    s = float(_pairdist_mean(cpk) / _pairdist_mean(cgk).clamp_min(1e-8))
    s = float(min(max(s, ALIGN_SCALE_CLAMP[0]), ALIGN_SCALE_CLAMP[1]))
    talign = cpk.mean(0) - s * (Ralign @ cgk.mean(0))
    if n_drop > 0:
        print(f"  [GT-ALIGN-ROBUST] dropped {n_drop}/{n} outlier predicted poses, re-estimated alignment")
    return s, Ralign, talign


def map_gt_pose_to_model(gt_E, gt_is_c2w, s, Ralign, talign, want_c2w):
    c, Rc2w = pose_to_center_Rc2w(gt_E, gt_is_c2w)
    c_m = s * (Ralign @ c) + talign
    Rc2w_m = Ralign @ Rc2w
    return (center_Rc2w_to_c2w(c_m, Rc2w_m) if want_c2w
            else center_Rc2w_to_w2c(c_m, Rc2w_m))


_align_print_count = 0
def debug_gt_align_print(model_tag, debug_tag, s, Ralign, talign,
                         gt_ctx_E, pred_ctx_E, gt_is_c2w, pred_is_c2w):
    global _align_print_count
    if _align_print_count >= DEBUG_ALIGN_PRINT_LIMIT: return
    with torch.no_grad():
        cg = torch.stack([pose_to_center_Rc2w(gt_ctx_E[i], gt_is_c2w)[0]
                          for i in range(gt_ctx_E.shape[0])])
        cp = torch.stack([pose_to_center_Rc2w(pred_ctx_E[i], pred_is_c2w)[0]
                          for i in range(pred_ctx_E.shape[0])])
        pred = s * (cg @ Ralign.T) + talign
        resid = (pred - cp).norm(dim=-1).mean().item()
        ref = cp.norm(dim=-1).mean().clamp_min(1e-8).item()
    print(f"  [GT-ALIGN] {model_tag} {debug_tag}: s={s:.4f} resid={resid:.4f} "
          f"rel={resid/ref:.3f}")
    if resid / ref > 0.25:
        print(f"           ! large GT<->pred alignment residual => poor model ctx "
              f"pose quality, or GT_EXTRINSIC_C2W convention is wrong")
    _align_print_count += 1


def front_inframe_frac(xyz, ext_w2c_3x4, K_pix, H, W):
    """xyz:(N,3) in the model frame. Returns (fraction in front of the camera,
    fraction in front AND projecting inside the image).
    in-frame ~ 0 -> alignment/pose points the camera away; in-frame high but the
    render is still empty -> gaussian orientation/scale, or this view was not reconstructed."""
    R = ext_w2c_3x4[:3, :3].float(); t = ext_w2c_3x4[:3, 3].float()
    cam = xyz.float() @ R.t() + t
    z = cam[:, 2]; front = z > 1e-4
    zc = z.clamp_min(1e-6)
    u = K_pix[0, 0] * cam[:, 0] / zc + K_pix[0, 2]
    v = K_pix[1, 1] * cam[:, 1] / zc + K_pix[1, 2]
    inframe = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return front.float().mean().item(), inframe.float().mean().item()


# ===========================================================
#  PLY export (optional)
# ===========================================================

def save_gaussians_ply(xyz, color, opa, path, opa_threshold=0.1):
    if isinstance(xyz, torch.Tensor):   xyz = xyz.detach().cpu().float()
    if isinstance(color, torch.Tensor): color = color.detach().cpu().float()
    if isinstance(opa, torch.Tensor):   opa = opa.detach().cpu().float()
    if opa.dim() == 2: opa = opa.squeeze(-1)
    mask = opa > opa_threshold
    xyz, color, opa = xyz[mask], color[mask].clamp(0, 1), opa[mask]
    n = xyz.shape[0]
    if n == 0:
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 0\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "property float opacity\nend_header\n")
        return 0
    xyz_np = xyz.numpy().astype(np.float32)
    color_np = (color.numpy() * 255).astype(np.uint8)
    opa_np = opa.numpy().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property float opacity\nend_header\n").encode('ascii'))
        buf = bytearray()
        for i in range(n):
            buf += struct.pack('<fff', xyz_np[i,0], xyz_np[i,1], xyz_np[i,2])
            buf += struct.pack('<BBB', color_np[i,0], color_np[i,1], color_np[i,2])
            buf += struct.pack('<f', opa_np[i])
        f.write(bytes(buf))
    return n


def export_gh_gaussians(pred_scene, save_dir, name_prefix):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    try:
        if EXPORT_GH_GAUSSIANS_PLY:
            save_gaussians_ply(pred_scene["xyz"], pred_scene["color"],
                pred_scene["opacity"],
                os.path.join(save_dir, f"{name_prefix}_gaussians.ply"), PLY_OPA_LOW)
        if EXPORT_GH_POINTCLOUD_PLY:
            save_gaussians_ply(pred_scene["xyz"], pred_scene["color"],
                pred_scene["opacity"],
                os.path.join(save_dir, f"{name_prefix}_pointcloud.ply"), PLY_OPA_HIGH)
    except Exception as e:
        print(f"  ! GH export failed {name_prefix}: {e}")


# ===========================================================
#  JSON-metadata driven: discover scenes + fetch GT poses + source-image paths
# ===========================================================

def remap_src(src):
    if SRC_DIR_REMAP is not None:
        old, new = SRC_DIR_REMAP
        if src.startswith(old): return new + src[len(old):]
    return src


def src_image_path(meta, fname):
    src = remap_src(meta["src_dir"])
    return os.path.join(src, meta.get("images_subdir", "images"), fname)


def find_scene_metas(dataset_dir, ctx_test_map, meta_glob="*.json"):
    """Robustly discover per-scene meta json.
    Two layouts are supported:
      - flat:            DATASET_DIR/<scene>.json
      - per-scene dir:   DATASET_DIR/<scene>/*.json
    The scene name comes from the json's scene_name field (not the dir/file name).
    Returns [(scene_name, meta), ...] sorted by scene name.
    """
    cand_paths = sorted(set(
        glob.glob(os.path.join(dataset_dir, meta_glob)) +        # flat
        glob.glob(os.path.join(dataset_dir, "*", meta_glob))     # per-scene dir
    ))
    seen = {}
    for c in cand_paths:
        try:
            with open(c) as f: m = json.load(f)
        except Exception:
            continue
        if not (isinstance(m, dict) and "configs" in m and "src_dir" in m):
            continue
        sname = m.get("scene_name") or os.path.splitext(os.path.basename(c))[0]
        if sname in seen:
            continue
        # check that every config exists + the source image is readable (from src_dir)
        ok = True
        for n in sorted(ctx_test_map.keys()):
            key = f"ctx_{n}"
            if key not in m["configs"]:
                print(f"  [skip] {sname}: meta missing {key}"); ok = False; break
            cfg = m["configs"][key]
            f0 = src_image_path(m, cfg["ctx"]["files"][0])
            if not os.path.isfile(f0):
                print(f"  [skip] {sname}: source image not found {f0} (check src_dir / SRC_DIR_REMAP)")
                ok = False; break
        if ok:
            seen[sname] = m
    if not seen:
        print(f"  ! no usable meta json found under {dataset_dir}")
        print(f"     tried: {dataset_dir}/*.json and {dataset_dir}/*/*.json")
        print(f"     if the json lives elsewhere, point DATASET_DIR at it")
    return sorted(seen.items())


def get_config_gt(meta, n, device):
    """Fetch the GT for one config: ctx/test (source-image paths, extrinsics(N,4,4), intrinsics(N,3,3), wh list)."""
    cfg = meta["configs"][f"ctx_{n}"]
    def _pack(side):
        d = cfg[side]
        paths = [src_image_path(meta, f) for f in d["files"]]
        E = torch.tensor(d["extrinsics"], dtype=torch.float32, device=device)   # (M,4,4)
        K = torch.tensor(d["intrinsics"], dtype=torch.float32, device=device)   # (M,3,3)
        wh = [tuple(x) for x in d["image_size"]]                                 # [(W,H),...]
        return paths, E, K, wh
    return _pack("ctx"), _pack("test")


# ===========================================================
#  Image saving (side-by-side / single)
# ===========================================================

def save_single(t, save_dir, fname):
    """Save a single render (for figures); silently skip if t is None, save_dir is
    blank, or anything fails, so it never affects evaluation."""
    if t is None or not save_dir: return
    try:
        os.makedirs(save_dir, exist_ok=True)
        arr = (t.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr.transpose(1, 2, 0)).save(os.path.join(save_dir, fname))
    except Exception:
        pass


def save_pair(r_ours, gt, psnr_ours, save_dir, name_stem):
    """Save a side-by-side (prediction | GT) panel."""
    if r_ours is None or not save_dir: return
    os.makedirs(save_dir, exist_ok=True)
    def to_pil(t):
        arr = (t.detach().cpu().float().clamp(0,1).numpy()*255).astype(np.uint8)
        return Image.fromarray(arr.transpose(1,2,0))
    panels = [(to_pil(r_ours), f"GaussianHead {psnr_ours:.2f}dB"), (to_pil(gt), "GT")]
    W_img, H_img = panels[0][0].width, panels[0][0].height
    n = len(panels)
    canvas = Image.new("RGB", (W_img*n + 4*(n-1), H_img+24), (30,30,30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, (im, lb) in enumerate(panels):
        x = i * (W_img + 4); canvas.paste(im, (x, 24))
        draw.text((x+2, 4), lb, fill=(255,230,60), font=font)
    canvas.save(os.path.join(save_dir, f"{name_stem}_psnr{psnr_ours:.2f}.png"))


# ===========================================================
#  GaussianHead: ctx-only forward -> render at the GT target pose
# ===========================================================

_printed_gaussians_keys = False
_printed_gh_intr = False
_self_check_count = {"gh": 0}

def gh_render_at_gt(model, ctx_src_paths, gt_ctx_E,
                    gt_test_E, gt_test_K, gt_test_wh,
                    device, dtype, eval_hw,
                    export_dir=None, name_prefix=None, debug_tag=""):
    """
    ctx-only forward -> gaussians + predicted ctx poses (w2c); align GT-ctx <->
    predicted-ctx; map the GT test pose into the GH frame and render at the eval
    resolution with the chosen intrinsics.
    Returns (rendered_eval(3,H,W), alpha_eval(H,W)).
    """
    global _printed_gaussians_keys
    H_e, W_e = eval_hw
    n_ctx_local = len(ctx_src_paths)          # timing group

    _cuda_sync(); _t_pre = time.time()        # timing: image read + preprocess
    imgs = load_and_preprocess_images(list(ctx_src_paths)).to(device)
    _cuda_sync(); _timing["gh_pre"].setdefault(n_ctx_local, []).append(time.time() - _t_pre)
    Hc, Wc = imgs.shape[-2:]
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        with _timed(_timing["gh_fwd"], n_ctx_local):   # timing: single forward (VGGT + Gaussian head)
            preds = model(imgs.unsqueeze(0))
    exts, itrs = pose_encoding_to_extri_intri(preds["pose_enc"], image_size_hw=(Hc, Wc))
    pred_ctx_E = exts[0].float()                                # (n,3,4) w2c
    pred_ctx_K = itrs[0].float().mean(0)                        # (3,3) proc resolution, shared across views

    if not _printed_gaussians_keys:
        print(f"\n  [one-off diag] preds['gaussians'] fields:")
        for k, v in preds["gaussians"].items():
            if isinstance(v, torch.Tensor):
                print(f"    {k:>18s}: {tuple(v.shape)} {v.dtype}")
        print()
        _printed_gaussians_keys = True

    pred_scene = {
        k: v[0].reshape(-1, v.shape[-1]).float()
        for k, v in preds["gaussians"].items()
        if isinstance(v, torch.Tensor) and v.dim() >= 2
    }
    del preds; torch.cuda.empty_cache()

    # GT(COLMAP) -> GH frame (both w2c)
    s, Ralign, talign = compute_gt_to_model_sim3(
        gt_ctx_E, pred_ctx_E, GT_EXTRINSIC_C2W, pred_is_c2w=False)
    debug_gt_align_print("GH", debug_tag, s, Ralign, talign,
                         gt_ctx_E, pred_ctx_E, GT_EXTRINSIC_C2W, False)

    if GH_USE_GT_INTRINSIC:
        K_pix = rescale_intrinsic_pixel(gt_test_K, gt_test_wh, eval_hw)
    else:
        K_pix = rescale_intrinsic_pixel(pred_ctx_K, (Wc, Hc), eval_hw)  # VGGT's own intrinsics -> eval
    global _printed_gh_intr
    if not _printed_gh_intr:
        Kg = rescale_intrinsic_pixel(gt_test_K, gt_test_wh, eval_hw)
        Kp = rescale_intrinsic_pixel(pred_ctx_K, (Wc, Hc), eval_hw)
        print(f"  [GH-INTR] eval(H,W)={eval_hw}  GT fx,fy={float(Kg[0,0]):.1f},{float(Kg[1,1]):.1f}"
              f"  VGGT fx,fy={float(Kp[0,0]):.1f},{float(Kp[1,1]):.1f}  -> rendering with "
              f"{'GT' if GH_USE_GT_INTRINSIC else 'predicted'} intrinsics (closer => predicted ~ GT)")
        _printed_gh_intr = True
    test_ext_m = map_gt_pose_to_model(
        gt_test_E, GT_EXTRINSIC_C2W, s, Ralign, talign, want_c2w=False)  # w2c 4x4
    test_ext_3x4 = test_ext_m[:3, :]

    if export_dir and name_prefix:
        export_gh_gaussians(pred_scene, export_dir, name_prefix)

    def _render(ext3x4, color):
        return render_gaussians(
            pred_scene["xyz"], pred_scene["rotation"], pred_scene["scale"],
            pred_scene["opacity"], color, ext3x4, K_pix, H_e, W_e)

    rendered = _render(test_ext_3x4, pred_scene["color"]).float().clamp(0, 1)
    # alpha (ones-color) -> white-background composite + coverage mask
    alpha_eval = None
    try:
        alpha = _render(test_ext_3x4, torch.ones_like(pred_scene["color"])).float().clamp(0, 1)
        if RENDER_BG_WHITE:
            rendered = composite_premult_on_white(rendered, alpha)
        alpha_eval = alpha.mean(0)
    except Exception as e:
        print(f"  ! [GH] alpha/white-bg failed: {e}")

    # blank-render diagnostic: when coverage is very low, distinguish
    # "pose points away" from "coverage gap"
    if alpha_eval is not None:
        cov_now = (alpha_eval > MASK_ALPHA_THR).float().mean().item()
        if cov_now < 0.2:
            ff, iff = front_inframe_frac(pred_scene["xyz"], test_ext_3x4, K_pix, H_e, W_e)
            print(f"  ! [GH-BLANK] {debug_tag}: cov={cov_now:.2f} front={ff:.2f} "
                  f"in-frame={iff:.2f} "
                  f"({'in-frame~0 => alignment/pose points away' if iff < 0.05 else 'in-frame ok => missing reconstruction / gaussian scale'})")

    # ctx GT-view self-check (pose is GT, so the error should be ~zero -> validates alignment + gaussians + precision)
    if DEBUG_SELF_CHECK and _self_check_count["gh"] < DEBUG_SELF_CHECK_LIMIT:
        _self_check_count["gh"] += 1
        try:
            ext0 = map_gt_pose_to_model(gt_ctx_E[0], GT_EXTRINSIC_C2W, s, Ralign,
                                        talign, want_c2w=False)[:3, :]
            r0 = _render(ext0, pred_scene["color"]).float().clamp(0, 1)
            ctx0 = resize_chw((imgs[0].float()*0.5+0.5).clamp(0,1)
                              if imgs.min() < -0.01 else imgs[0].float().clamp(0,1), eval_hw)
            print(f"  [GT-SELF] GH {debug_tag}: ctx0@GT={compute_psnr(r0, ctx0):.2f}dB "
                  f"(low => alignment/gaussian/convention issue, try flipping GT_EXTRINSIC_C2W)")
        except Exception as e:
            print(f"  [GT-SELF] GH self-check failed: {e}")

    del pred_scene; torch.cuda.empty_cache()
    return rendered.float().clamp(0, 1), alpha_eval


# ===========================================================
#  Single (scene, ctx) evaluation
# ===========================================================

def _coverage_mask(gh_alpha, eval_hw):
    """Coverage mask from the GaussianHead alpha (the region the model actually filled)."""
    if gh_alpha is not None:
        return gh_alpha > MASK_ALPHA_THR
    return torch.ones(eval_hw, dtype=torch.bool)


def evaluate_one(gh_model, scene_name, meta, n_ctx, device, dtype,
                 save_img_dir, save_ply_dir):
    (ctx_paths, ctx_E, ctx_K, ctx_wh), (test_paths, test_E, test_K, test_wh) = \
        get_config_gt(meta, n_ctx, device)

    gh_p, gh_s, gh_l, gh_pm, gh_sm, gh_cov = [], [], [], [], [], []

    for ti, test_p in enumerate(test_paths):
        stem = f"ctx{n_ctx}_{scene_name}_test{ti:02d}"
        export_dir = save_ply_dir if ti == 0 else None
        gh_pref  = f"ctx{n_ctx}_{scene_name}_gh" if ti == 0 else None

        gt, orig_wh, eval_hw = load_canonical_gt_with_size(test_p)
        gt = gt.to(device)
        gt_test_E = test_E[ti]
        gt_test_K = test_K[ti]
        gt_test_wh = test_wh[ti]
        dbg = f"{scene_name} ctx{n_ctx} t{ti}"

        r_ours = None; gh_alpha = None
        try:
            r_ours, gh_alpha = gh_render_at_gt(
                gh_model, ctx_paths, ctx_E, gt_test_E, gt_test_K, gt_test_wh,
                device, dtype, eval_hw, export_dir, gh_pref, dbg)
            p_o = compute_psnr(r_ours, gt); s_o = compute_ssim(r_ours, gt)
            l_o = compute_lpips(r_ours, gt, device)
            gh_p.append(p_o); gh_s.append(s_o)
            if l_o >= 0: gh_l.append(l_o)
            if gh_alpha is not None:
                gh_cov.append((gh_alpha > MASK_ALPHA_THR).float().mean().item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); continue
            raise

        # masked metrics inside the coverage mask
        if EVAL_MASKED and r_ours is not None:
            mask = _coverage_mask(gh_alpha, eval_hw)
            pm = compute_psnr_masked(r_ours, gt, mask); sm = compute_ssim_masked(r_ours, gt, mask)
            if not math.isnan(pm): gh_pm.append(pm)
            if not math.isnan(sm): gh_sm.append(sm)

        try:
            save_pair(r_ours, gt, compute_psnr(r_ours, gt), save_img_dir, stem)
            save_single(r_ours, SAVE_DIR_GH_IMG, f"{stem}.png")    # single prediction image (no psnr in name)
        except Exception:
            pass
        torch.cuda.empty_cache()

    def avg(x): return sum(x)/len(x) if x else float("nan")
    return (avg(gh_p), avg(gh_s), avg(gh_l), avg(gh_pm), avg(gh_sm), avg(gh_cov))


# ===========================================================
#  Main
# ===========================================================

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if sm >= 8 else torch.float16

    for d in (SAVE_DIR_IMGS, SAVE_DIR_PLY, SAVE_DIR_GH_IMG):
        if d: os.makedirs(d, exist_ok=True)
    if LOG_FILE and os.path.dirname(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print("="*72)
    print("  Novel-view synthesis eval (GT poses + GT-anchored alignment)")
    print("="*72)
    print(f"  Dataset meta : {DATASET_DIR}")
    print(f"  Ckpt         : {CKPT_PATH}")
    print(f"  ctx configs  : " + ", ".join(f"ctx{n}+test{CTX_TEST_MAP[n]}" for n in CTX_COUNTS))
    print(f"  GT extrinsic : {'c2w' if GT_EXTRINSIC_C2W else 'w2c'}")
    print(f"  Eval res     : long side {EVAL_LONG_SIDE}px, canonical GT (from src images)")
    print(f"  Masked       : coverage mask (thr={MASK_ALPHA_THR})")
    print("="*72)

    print("\n[1/3] Loading VGGT + GaussianHead ...")
    _cuda_sync(); _t_load = time.time()       # timing: VGGT+GH load (one-off)
    gh_model = VGGT.from_pretrained("facebook/VGGT-1B", enable_gaussian=True).to(device)
    assert os.path.isfile(CKPT_PATH), f"CKPT_PATH not found: '{CKPT_PATH}' (set CKPT_PATH)"
    ckpt = torch.load(CKPT_PATH, map_location=device)
    gh_model.gaussian_head.load_state_dict(ckpt["gaussian_head_state_dict"], strict=False)
    if "dpt_feature_head_state_dict" in ckpt:
        gh_model.dpt_feature_head.load_state_dict(ckpt["dpt_feature_head_state_dict"], strict=False)
    gh_model.eval()
    _cuda_sync(); _timing["gh_load"] = time.time() - _t_load
    print(f"  step={ckpt.get('global_step',0)} "
          f"best PSNR={ckpt.get('metrics',{}).get('psnr',0):.2f} "
          f"(VGGT+GH load {_timing['gh_load']:.1f}s)")

    print("\n[2/3] Scanning meta.json ...")
    metas = find_scene_metas(DATASET_DIR, CTX_TEST_MAP, META_GLOB)
    if not metas: raise RuntimeError(f"No usable meta under {DATASET_DIR}")
    if len(metas) > N_SCENES: metas = metas[:N_SCENES]
    print(f"  {len(metas)} scenes: " + ", ".join(s for s,_ in metas))

    print("\n[3/3] Evaluating ...")
    t0 = time.time()
    summary = {}
    rows_by_ctx = {}

    for n_ctx in sorted(CTX_COUNTS):
        print(f"\n  --- ctx={n_ctx} ---")
        gh_ps, gh_ss, gh_ls, gh_pms, gh_sms, gh_cv = [],[],[],[],[],[]
        rows = []
        for sname, meta in metas:
            tsc = time.time()
            try:
                gh_m = evaluate_one(gh_model, sname, meta, n_ctx, device, dtype,
                                    SAVE_DIR_IMGS, SAVE_DIR_PLY)
            except Exception as e:
                print(f"  ! {sname}: {type(e).__name__}: {e}")
                traceback.print_exc(); torch.cuda.empty_cache(); continue
            gp,gs,gl,gpm,gsm,gcov = gh_m
            dur = format_dur(time.time()-tsc)
            if not math.isnan(gp):
                gh_ps.append(gp); gh_ss.append(gs)
                if gl>=0: gh_ls.append(gl)
            for L,v in ((gh_pms,gpm),(gh_sms,gsm),(gh_cv,gcov)):
                if not math.isnan(v): L.append(v)
            print(f"  {sname:<10s} gh {gp:6.2f}/{gpm:6.2f}m cov{gcov:.2f}  ({dur})")
            rows.append((sname,gp,gs,gl,gcov,gpm,gsm))
            torch.cuda.empty_cache()

        def avg(x): return sum(x)/len(x) if x else float("nan")
        summary[n_ctx] = {
            "n": len(gh_ps),
            "gh_full":(avg(gh_ps),avg(gh_ss),avg(gh_ls)), "gh_cov":avg(gh_cv),
            "gh_mask":(avg(gh_pms),avg(gh_sms)),
        }
        rows_by_ctx[n_ctx] = rows

    tt = format_dur(time.time()-t0)

    print("\n" + "="*80)
    print("  Summary — novel-view synthesis at GT poses")
    print("="*80)
    print(f"  {'ctx':>4} {'method':<16s} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'cov%':>6} {'PSNR_m':>7} {'SSIM_m':>7}")
    print("  " + "-"*72)
    for n in sorted(CTX_COUNTS):
        s = summary[n]
        gp,gs,gl = s["gh_full"]; gpm,gsm = s["gh_mask"]
        print(f"  {n:>4} {'GaussianHead':<16s} {gp:7.2f} {gs:7.4f} {gl:7.4f} "
              f"{100*s['gh_cov']:6.1f} {gpm:7.2f} {gsm:7.4f}")
        print("  " + "-"*72)
    print("  full=whole image; cov%=fraction of the test frame filled;")
    print("  PSNR_m/SSIM_m=inside the coverage mask (fidelity of the reconstructed region).")

    # ========================================================
    #  Forward / load timing
    # ========================================================
    def _fmt_load(x): return f"{x:.1f}s" if x is not None else "N/A"
    ctxs_t = sorted(_timing["gh_fwd"].keys())

    print("\n" + "="*80)
    print("  Forward / load timing (CUDA-synchronized)")
    print("="*80)
    print(f"  Model load (one-off, amortized over all scenes): VGGT+GH={_fmt_load(_timing['gh_load'])}")
    print(f"  Single forward (pure network inference, model(...)):")
    print(f"    {'ctx':>4} {'median':>10} {'fastest':>10} {'n':>6}")
    print("    " + "-"*34)
    for n in ctxs_t:
        gmed,gmin,_,gn = _fwd_stats(_timing["gh_fwd"].get(n, []))
        print(f"    {n:>4} {gmed:>9.3f}s {gmin:>9.3f}s {gn:>6}")
    gmed,gmin,_,gn = _fwd_stats(_flatten(_timing["gh_fwd"]))
    print(f"    {'all':>4} {gmed:>9.3f}s {gmin:>9.3f}s {gn:>6}")
    gpre = _fwd_stats(_flatten(_timing["gh_pre"]))[0]
    print(f"  Image read + preprocess (median, not counted as forward): {gpre:.3f}s")
    print("  Note: median/fastest exclude the first CUDA warmup; forward time grows with #ctx views.")

    print(f"\n  Total: {tt}")
    if SAVE_DIR_IMGS: print(f"  Side-by-side images: {SAVE_DIR_IMGS}")

    if LOG_FILE:
        with open(LOG_FILE, "w") as f:
            f.write(f"eval_nvs  ctx={CTX_COUNTS}  scenes={len(metas)}  {tt}\n")
            f.write(f"ckpt: {CKPT_PATH}\n")
            f.write("protocol: test views never input; rendered at GT poses (GT extrinsics + own intrinsics);\n")
            f.write("          GT-ctx<->predicted-ctx alignment (orientation->rotation / baseline->scale);\n")
            f.write("          full + coverage mask + coverage fraction\n")
            f.write(f"          GT_EXTRINSIC_C2W={GT_EXTRINSIC_C2W}\n\n")
            for n in sorted(CTX_COUNTS):
                s = summary[n]; gp,gs,gl = s["gh_full"]; gpm,gsm = s["gh_mask"]
                f.write(f"=== ctx={n} ({s['n']} scenes) ===\n")
                f.write(f"[full] PSNR={gp:.4f} SSIM={gs:.4f} LPIPS={gl:.4f} cov={100*s['gh_cov']:.1f}%\n")
                f.write(f"[mask] PSNR={gpm:.4f} SSIM={gsm:.4f}\n\n")

            f.write("===== forward / load timing (CUDA-synchronized) =====\n")
            f.write(f"model load (one-off): VGGT+GH={_fmt_load(_timing['gh_load'])}\n")
            f.write("single forward (pure network inference, s):\n")
            f.write(f"  {'ctx':>4} {'median':>10} {'min':>9} {'n':>6}\n")
            for n in ctxs_t:
                gmed,gmin,gmean,gn = _fwd_stats(_timing["gh_fwd"].get(n, []))
                f.write(f"  {n:>4} {gmed:>10.4f} {gmin:>9.4f} {gn:>6}\n")
            gmed,gmin,gmean,gn = _fwd_stats(_flatten(_timing["gh_fwd"]))
            f.write(f"  {'all':>4} {gmed:>10.4f} {gmin:>9.4f} {gn:>6}\n")
            f.write(f"image read + preprocess (median): {gpre:.4f}s\n")
            f.write("(median/fastest exclude the first CUDA warmup; forward time grows with #ctx views)\n\n")

            for n in sorted(CTX_COUNTS):
                f.write(f"\n===== ctx={n} per-scene =====\n")
                f.write(f"  {'scene':<10s} {'PSNR':>7}{'cov':>7}  {'PSNR_m':>9}\n")
                for (sn,gp,gs,gl,gcov,gpm,gsm) in rows_by_ctx[n]:
                    gm = f"{gpm:9.3f}" if not math.isnan(gpm) else f"{'N/A':>9}"
                    f.write(f"  {sn:<10s} {gp:7.3f}{100*gcov:7.1f}  {gm}\n")
        print(f"\n  Log: {LOG_FILE}")


if __name__ == "__main__":
    main()