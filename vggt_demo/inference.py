
import os
import threading

import numpy as np

from vggt_demo import exporters

# Important: same as training, xyz_base uses depth unprojection
# (must be set before importing/using vggt).
os.environ.setdefault("VGGT_XYZ_BASE_SOURCE", "depth_unproject")

# ---- config (relative-path defaults, overridable via env vars) ----
DEFAULT_CKPT = "output/nogt_geoinit_pure/gaussian_head_best_test_nogt.pth"
HF_MODEL_ID = os.environ.get("VGGT_HF_MODEL", "facebook/VGGT-1B")


def get_ckpt_path():
    p = os.environ.get("CKPT_PATH", DEFAULT_CKPT)
    if not os.path.isabs(p):
        # resolve relative paths against the repo root (app.py writes the detected
        # repo root into VGGT_REPO_ROOT)
        root = os.environ.get("VGGT_REPO_ROOT", os.getcwd())
        p = os.path.join(root, p)
    return p


_MODEL = None
_DEVICE = None
_DTYPE = None
_LOAD_LOCK = threading.Lock()
_LOAD_ERROR = None  # records the load-failure reason for display in the UI


def _torch_cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def decide_mode():
    """Return ('real'|'mock', reason_str)."""
    if os.environ.get("DEMO_MOCK", "0") == "1":
        return "mock", "Forced MOCK (DEMO_MOCK=1)"
    try:
        import torch  # noqa: F401
    except Exception as e:
        return "mock", f"torch not installed ({e})"
    if not _torch_cuda_ok():
        return "mock", "No CUDA available (VGGT-1B is not suited to CPU inference)"
    ckpt = get_ckpt_path()
    if not os.path.exists(ckpt):
        return "mock", f"checkpoint not found: {ckpt}"
    if _LOAD_ERROR is not None:
        return "mock", f"model load failed: {_LOAD_ERROR}"
    return "real", "REAL (CUDA + checkpoint ready)"


def _build_and_load_model():
    """Build VGGT and load weights (mirrors train_nogt.py). Thread-safe; loads once."""
    global _MODEL, _DEVICE, _DTYPE, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        import torch
        from vggt.models.vggt import VGGT

        device = "cuda"
        cap = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16

        print(f"[demo] loading VGGT ({HF_MODEL_ID}) ... device={device} dtype={dtype}", flush=True)
        model = VGGT.from_pretrained(HF_MODEL_ID, enable_gaussian=True).to(device)

        # inference: single-frame chunk, no gradient checkpointing
        if hasattr(model.gaussian_head, "frames_chunk_size"):
            model.gaussian_head.frames_chunk_size = 1
        if hasattr(model.gaussian_head, "use_checkpoint"):
            model.gaussian_head.use_checkpoint = False

        # training used ENABLE_COLOR_HEAD=True -> color_head must exist before loading,
        # otherwise the state_dict keys won't match
        if getattr(model.gaussian_head, "color_head", None) is None:
            if hasattr(model.gaussian_head, "enable_color_head_after_init"):
                model.gaussian_head.enable_color_head_after_init(device=device)

        ckpt_path = get_ckpt_path()
        print(f"[demo] loading checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.gaussian_head.load_state_dict(ckpt["gaussian_head_state_dict"])
        if "dpt_feature_head_state_dict" in ckpt:
            model.dpt_feature_head.load_state_dict(ckpt["dpt_feature_head_state_dict"])

        model.eval()
        _MODEL, _DEVICE, _DTYPE = model, device, dtype
        m = ckpt.get("metrics", {})
        if m:
            print(f"[demo] ckpt metrics: {m}", flush=True)
        print("[demo] model ready", flush=True)
        return _MODEL


def _run_real(image_paths):
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images

    model = _build_and_load_model()
    device, dtype = _DEVICE, _DTYPE

    images = load_and_preprocess_images(list(image_paths)).to(device)  # (S,3,H,W) in [0,1]
    S, _, H, W = images.shape
    images_input = images.unsqueeze(0)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images_input)

    g = preds["gaussians"]

    def merge(key):
        v = g[key]                       # (1, S, N, D)
        return v[0].reshape(-1, v.shape[-1]).float().cpu().numpy()

    out = {
        "xyz": merge("xyz"),
        "color": merge("color"),
        "opacity": merge("opacity"),
        "scale": merge("scale"),
        "rot": merge("rotation"),
        "meta": {"num_views": int(S), "H": int(H), "W": int(W),
                 "num_gaussians": int(S * H * W), "mode": "real"},
    }
    del preds, images_input, images
    torch.cuda.empty_cache()
    return out


def _run_mock(image_paths):
    n = 60000
    xyz, color, opacity, scale, rot = exporters.make_mock_gaussians(n=n, seed=0)
    return {
        "xyz": xyz, "color": color, "opacity": opacity, "scale": scale, "rot": rot,
        "meta": {"num_views": len(list(image_paths)), "H": 0, "W": 0,
                 "num_gaussians": n, "mode": "mock"},
    }


def run_inference(image_paths):
    """Main entry. On failure, fall back to mock and record the reason in meta['warn']."""
    global _LOAD_ERROR
    mode, _ = decide_mode()
    if mode == "real":
        try:
            return _run_real(image_paths)
        except Exception as e:
            _LOAD_ERROR = repr(e)
            print(f"[demo] REAL inference failed, falling back to MOCK: {e}", flush=True)
            out = _run_mock(image_paths)
            out["meta"]["warn"] = f"REAL failed, fell back to MOCK: {e}"
            return out
    return _run_mock(image_paths)