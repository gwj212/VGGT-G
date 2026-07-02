# Run (frontend validation, any machine, no GPU/checkpoint -> auto MOCK):
#     pip install -r requirements-demo.txt
#     python app.py
#
# Run (real inference, needs CUDA + checkpoint + vggt deps installed):
#     python app.py

# ============================================================

import os
import sys
import uuid
import base64

import numpy as np

# ============================ USER CONFIG ============================
# Image-upload feature switch:
#   ENABLE_UPLOAD = False  ->  upload button disabled (OFF)   [default]
#   ENABLE_UPLOAD = True   ->  upload button enabled  (ON)
# (Optional: override at launch without editing -> env DEMO_ENABLE_UPLOAD=1)
ENABLE_UPLOAD = False
if os.environ.get("DEMO_ENABLE_UPLOAD") is not None:
    ENABLE_UPLOAD = (os.environ["DEMO_ENABLE_UPLOAD"] == "1")
# =====================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """Walk up from `start` to find the dir containing the vggt/ package."""
    d = os.path.abspath(start)
    for _ in range(8):
        if (os.path.exists(os.path.join(d, "vggt", "models", "vggt.py"))
                or os.path.exists(os.path.join(d, "vggt", "__init__.py"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


REPO_ROOT = (os.environ.get("VGGT_REPO_ROOT")
             or _find_repo_root(_HERE)
             or os.getcwd())
REPO_ROOT = os.path.abspath(REPO_ROOT)
os.environ["VGGT_REPO_ROOT"] = REPO_ROOT          # so inference can resolve the relative checkpoint path

for _p in (REPO_ROOT, _HERE):                      # make `import vggt` and `import vggt_demo` both work
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr

from vggt_demo import exporters
from vggt_demo import examples_util
from vggt_demo.inference import run_inference, decide_mode, get_ckpt_path


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


OUTPUT_DIR = _resolve(
    os.environ.get("DEMO_OUTPUT_DIR", os.path.join(_HERE, "demo_outputs")), REPO_ROOT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ex_env = os.environ.get("DEMO_EXAMPLES_DIR")
EXAMPLES_DIR = (_resolve(_ex_env, REPO_ROOT) if _ex_env
                else os.path.join(_HERE, "demo_assets", "examples"))

PC_VIEW = "Point cloud (fast, free pan)"
SPLAT_VIEW = "Gaussian splat (continuous)"

print(f"[demo] REPO_ROOT   = {REPO_ROOT}", flush=True)
print(f"[demo] CKPT_PATH   = {get_ckpt_path()}", flush=True)
print(f"[demo] EXAMPLES    = {EXAMPLES_DIR}", flush=True)

SCENES = examples_util.list_scenes(EXAMPLES_DIR)
SCENE_NAMES = list(SCENES.keys())
_MODE, _MODE_REASON = decide_mode()


# --------------------------- helpers ---------------------------

def _scene_gallery(scene_name):
    if not scene_name or scene_name not in SCENES:
        return []
    return SCENES[scene_name]


def _scene_num_images(scene_name):
    """How many images the given scene has (0 if unknown)."""
    if not scene_name or scene_name not in SCENES:
        return 0
    return len(SCENES[scene_name])


def _paths_from_upload(upload):
    """Extract file paths from an upload value (handles str / objects with .name)."""
    if not upload:
        return []
    items = upload if isinstance(upload, (list, tuple)) else [upload]
    paths = []
    for it in items:
        if it is None:
            continue
        p = it if isinstance(it, str) else getattr(it, "name", None)
        if p:
            paths.append(p)
    return paths


def _even_subsample(paths, n):
    """Pick n images evenly from paths (first/last included). n<=0 or n>=total -> all."""
    n = int(n)
    L = len(paths)
    if n <= 0 or n >= L:
        return list(paths)
    idx = np.linspace(0, L - 1, n).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    return [paths[i] for i in idx]


def _fmt_stats(meta, kept, total_after_filter, preview_style):
    mode = meta.get("mode", "?")
    lines = []
    badge = "🟢 REAL (real inference)" if mode == "real" else "🟡 MOCK (synthetic data - frontend validation only)"
    lines.append(f"**Mode**: {badge}")
    if meta.get("warn"):
        lines.append(f"> ⚠️ {meta['warn']}")
    if mode == "real":
        lines.append(f"**Input views**: {meta['num_views']} - resolution {meta['H']}x{meta['W']}")
    else:
        lines.append(f"**Input views**: {meta['num_views']} (mock ignores image content)")
    lines.append(f"**Raw Gaussians**: {meta['num_gaussians']:,}")
    lines.append(f"**After opacity filter**: {total_after_filter:,}")
    lines.append(f"**Exported / shown**: {kept:,}")
    if preview_style == SPLAT_VIEW:
        lines.append("> Splat preview: drag to rotate freely (any angle), wheel zoom, right-drag pan.")
    return "\n\n".join(lines)


_PC_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:__BG__;overflow:hidden;font-family:sans-serif}
#c{width:100%;height:100%;display:block;outline:none}
#ui{position:absolute;top:8px;left:8px;color:#ddd;font-size:12px;background:rgba(0,0,0,.45);padding:6px 8px;border-radius:6px}
#ui label{margin-right:4px}
#hint{position:absolute;bottom:8px;left:8px;color:#9aa;font-size:11px;background:rgba(0,0,0,.45);padding:4px 8px;border-radius:6px}
#err{position:absolute;top:40%;width:100%;text-align:center;color:#c0392b;font-size:13px}
button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:2px 8px;cursor:pointer;margin-left:6px}
</style></head><body>
<canvas id="c"></canvas>
<div id="ui"><label>Point size</label><input id="ps" type="range" min="0.2" max="8" step="0.1" value="1"><button id="reset">Reset view</button></div>
<div id="hint">Left-drag = rotate (any angle) - Right-drag = pan - Wheel = zoom</div>
<div id="err"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>
<script>
try{
const N=__N__, RAD=__RAD__, CX=__CX__, CY=__CY__, CZ=__CZ__, BASE=__BASE__;
function b64(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const posU8=b64("__POS_B64__"), colU8=b64("__COL_B64__");
const positions=new Float32Array(posU8.buffer);
const colors=new Float32Array(N*3);
for(let i=0;i<N*3;i++){colors[i]=colU8[i]/255;}
function discTex(){const s=64,cv=document.createElement('canvas');cv.width=cv.height=s;const x=cv.getContext('2d');
  const g=x.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);g.addColorStop(0,'rgba(255,255,255,1)');g.addColorStop(0.5,'rgba(255,255,255,0.9)');g.addColorStop(1,'rgba(255,255,255,0)');
  x.fillStyle=g;x.fillRect(0,0,s,s);const t=new THREE.CanvasTexture(cv);return t;}
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true});
renderer.setPixelRatio(window.devicePixelRatio||1);
renderer.setClearColor(new THREE.Color('__BG__'),1.0);
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(50,1,Math.max(RAD*0.001,1e-4),RAD*200+10);
// mip-nerf360 / COLMAP scenes are commonly Y-down; start the up vector at -Y so
// it doesn't appear upside-down. With TrackballControls you can still roll freely.
cam.up.set(0,-1,0);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
const mat=new THREE.PointsMaterial({size:BASE,map:discTex(),vertexColors:true,transparent:true,alphaTest:0.02,depthWrite:true,sizeAttenuation:true});
const pts=new THREE.Points(geo,mat); scene.add(pts);
// TrackballControls: no polar clamp, no fixed up-axis -> rotate to ANY angle, never stuck.
const controls=new THREE.TrackballControls(cam,renderer.domElement);
controls.rotateSpeed=3.0; controls.zoomSpeed=1.2; controls.panSpeed=0.8;
controls.noZoom=false; controls.noPan=false; controls.staticMoving=false; controls.dynamicDampingFactor=0.15;
function frameView(){controls.target.set(CX,CY,CZ);cam.position.set(CX,CY,CZ+RAD*2.6);cam.up.set(0,-1,0);controls.update();}
frameView();
function resize(){const w=canvas.clientWidth||1,h=canvas.clientHeight||1;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();controls.handleResize();}
window.addEventListener('resize',resize); resize();
document.getElementById('ps').addEventListener('input',e=>{mat.size=BASE*parseFloat(e.target.value);});
document.getElementById('reset').addEventListener('click',frameView);
(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,cam);})();
}catch(e){document.getElementById('err').textContent='Viewer init failed: '+e.message+' (offline environments may be unable to load the three.js CDN)';}
</script></body></html>"""


# -- inline Gaussian-splat viewer: continuous splats (gsplat.js, loaded in-browser) --
_SPLAT_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:__BG__;overflow:hidden;font-family:sans-serif}
#c{width:100%;height:100%;display:block;outline:none}
#hint{position:absolute;bottom:8px;left:8px;color:#9aa;font-size:11px;background:rgba(0,0,0,.45);padding:4px 8px;border-radius:6px}
#err{position:absolute;top:42%;width:100%;text-align:center;color:#e88;font-size:13px;padding:0 12px}
</style></head><body>
<canvas id="c"></canvas>
<div id="hint">Left-drag = rotate - Wheel = zoom (continuous Gaussian splatting)</div>
<div id="err"></div>
<script type="module">
const err=document.getElementById('err');
try{
  const SPLAT = await import("https://cdn.jsdelivr.net/npm/gsplat@latest/+esm");
  function b64(s){const r=atob(s);const u=new Uint8Array(r.length);for(let i=0;i<r.length;i++)u[i]=r.charCodeAt(i);return u;}
  const canvas=document.getElementById('c');
  const renderer=new SPLAT.WebGLRenderer(canvas);
  const scene=new SPLAT.Scene();
  const camera=new SPLAT.Camera();
  const controls=new SPLAT.OrbitControls(camera, canvas);
  const data=b64("__SPLAT_B64__");
  const splat=new SPLAT.Splat(SPLAT.SplatData.Deserialize(data));
  scene.addObject(splat);
  function resize(){renderer.setSize(canvas.clientWidth||1, canvas.clientHeight||1);}
  window.addEventListener('resize',resize); resize();
  const loop=()=>{controls.update();renderer.render(scene,camera);requestAnimationFrame(loop);};
  requestAnimationFrame(loop);
}catch(e){err.textContent='Splat viewer failed: '+e.message+' (needs CDN access; otherwise switch the preview to Point cloud)';}
</script></body></html>"""


def build_pointcloud_viewer_html(xyz, color, height=520, point_size=None, bg="#ffffff"):
    xyz = np.asarray(xyz, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, color = xyz[finite], color[finite]
    n = xyz.shape[0]
    if n == 0:
        return "<div style='color:#e88'>No points to display.</div>"
    c = xyz.mean(axis=0)
    rad = max(float(np.linalg.norm(xyz - c, axis=1).max()), 1e-3)
    if point_size is None:
        point_size = rad * 0.012   # bigger default so the surface reads as continuous
    pos_b64 = base64.b64encode(np.ascontiguousarray(xyz, dtype="<f4").tobytes()).decode("ascii")
    col_u8 = (np.clip(color, 0, 1) * 255 + 0.5).astype(np.uint8)
    col_b64 = base64.b64encode(np.ascontiguousarray(col_u8).tobytes()).decode("ascii")
    html = (_PC_TEMPLATE
            .replace("__N__", str(n)).replace("__RAD__", f"{rad:.6f}")
            .replace("__CX__", f"{float(c[0]):.6f}").replace("__CY__", f"{float(c[1]):.6f}")
            .replace("__CZ__", f"{float(c[2]):.6f}").replace("__BASE__", f"{float(point_size):.6f}")
            .replace("__BG__", bg).replace("__POS_B64__", pos_b64).replace("__COL_B64__", col_b64))
    esc = html.replace("&", "&amp;").replace('"', "&quot;")
    return (f'<iframe srcdoc="{esc}" style="width:100%;height:{int(height)}px;border:0;'
            f'border-radius:8px;overflow:hidden;background:{bg}"></iframe>')


def build_splat_viewer_html(splat_bytes, height=520, bg="#050507"):
    b = base64.b64encode(splat_bytes).decode("ascii")
    html = _SPLAT_TEMPLATE.replace("__BG__", bg).replace("__SPLAT_B64__", b)
    esc = html.replace("&", "&amp;").replace('"', "&quot;")
    return (f'<iframe srcdoc="{esc}" style="width:100%;height:{int(height)}px;border:0;'
            f'border-radius:8px;overflow:hidden;background:{bg}"></iframe>')


def _render_preview(cache, preview_style, scale_mult):
    """Build the inline viewer HTML from cached Gaussians, applying scale_mult.

    Used both by run() (first render) and by the scale slider (re-render only,
    no re-inference)."""
    vx = cache["xyz"]; vc = cache["color"]
    vo = cache["opacity"]; vs = cache["scale"]; vr = cache["rot"]
    sm = float(scale_mult)
    if preview_style == SPLAT_VIEW:
        # normalize to a unit-size object centered at origin so gsplat frames it
        c = vx.mean(axis=0)
        rad = max(float(np.linalg.norm(vx - c, axis=1).max()), 1e-3)
        nb = exporters.splat_buffer((vx - c) / rad, vc, vo, vs / rad, vr,
                                    point_cloud_look=False, scale_mult=sm)
        return build_splat_viewer_html(nb, height=520)
    else:
        # point-cloud preview: scale_mult scales the point size
        c = vx.mean(axis=0)
        rad = max(float(np.linalg.norm(vx - c, axis=1).max()), 1e-3)
        return build_pointcloud_viewer_html(vx, vc, height=520,
                                            point_size=rad * 0.012 * sm)


def restyle(cache, preview_style, scale_mult):
    """Re-render the preview when the scale slider (or preview style) changes,
    reusing the cached Gaussians. Does NOT re-run the model."""
    if not cache:
        return gr.update()  # nothing reconstructed yet; leave the viewer as-is
    return _render_preview(cache, preview_style, scale_mult)


def run(scene_name, uploaded, num_views, preview_style, opacity_thr, max_points, scale_mult):
    up_paths = _paths_from_upload(uploaded) if ENABLE_UPLOAD else []
    if up_paths:
        image_paths = up_paths
        src = "uploaded images"
    elif scene_name:
        image_paths = SCENES.get(scene_name, [])
        src = f"example: {scene_name}"
    else:
        raise gr.Error("Upload images, or pick an example scene on the left.")
    if not image_paths:
        raise gr.Error("No usable input images.")

    n_avail = len(image_paths)
    image_paths = _even_subsample(image_paths, num_views)
    n_used = len(image_paths)
    if n_used < 1:
        raise gr.Error("0 usable images.")

    out = run_inference(image_paths)
    xyz, color = out["xyz"], out["color"]
    opacity, scale, rot = out["opacity"], out["scale"], out["rot"]
    meta = out["meta"]
    total_raw = xyz.shape[0]

    thr = float(opacity_thr)
    keep_mask = (opacity.reshape(-1) >= thr) if thr > 0 else None
    total_after_filter = int(keep_mask.sum()) if keep_mask is not None else total_raw

    fxyz, fcolor, fopa, fscale, frot, _idx = exporters.filter_and_subsample(
        xyz, color, opacity, scale, rot,
        opacity_threshold=thr, max_points=int(max_points), seed=0,
    )
    kept = fxyz.shape[0]
    if kept == 0:
        raise gr.Error("No Gaussians left after filtering; lower the opacity threshold.")

    tag = uuid.uuid4().hex[:8]
    sm = float(scale_mult)
    # downloads keep the raw VGGT world coordinates (standard convention)
    gs_ply = os.path.join(OUTPUT_DIR, f"gaussian_splat_{tag}.ply")
    exporters.write_3dgs_ply(gs_ply, fxyz, fcolor, fopa, fscale, frot,
                             point_cloud_look=False, scale_mult=sm)
    pc_ply = os.path.join(OUTPUT_DIR, f"point_cloud_{tag}.ply")
    exporters.write_pointcloud_ply(pc_ply, fxyz, fcolor)

    # inline preview: show all kept points (no extra cap; the Max-points slider controls it)
    vx, vc, vo, vs, vr = fxyz, fcolor, fopa, fscale, frot

    # cache the filtered Gaussians so the scale slider can re-render without re-running the model
    cache = {"xyz": vx, "color": vc, "opacity": vo, "scale": vs, "rot": vr,
             "meta": meta, "kept": kept, "total_after_filter": total_after_filter,
             "src": src, "n_avail": n_avail, "n_used": n_used}

    viewer_html_val = _render_preview(cache, preview_style, sm)

    stats = _fmt_stats(meta, kept, total_after_filter, preview_style)
    head = f"**Source**: {src} | {n_avail} available -> using **{n_used}**"
    if n_used == 1:
        head += " (only 1 image; limited multi-view effect, >=3 recommended)"
    stats = head + "\n\n" + stats
    return viewer_html_val, image_paths, stats, gs_ply, pc_ply, cache


# --------------------------- UI ---------------------------

_DESC = f"""
# VGGT-Gaussian: Feed-Forward 3D Gaussian Reconstruction

Select a set of **multi-view images**; the model predicts a 3D Gaussian per pixel in a
single feed-forward pass. The merged result can be viewed and downloaded as a
**Gaussian Splat** or a **colored point cloud**. No camera poses, no per-scene optimization.

> Current mode: **{'🟢 REAL' if _MODE == 'real' else '🟡 MOCK'}** - {_MODE_REASON}
"""

_NO_SCENE_TIP = """
> ⚠️ No example scenes detected. Put subdirectories under `demo_assets/examples/`,
> one per scene, each containing several images of the same object/scene.
"""

try:
    _GR_MAJOR = int(str(gr.__version__).split(".")[0])
except Exception:
    _GR_MAJOR = 0
_THEME = gr.themes.Soft()
_blocks_kwargs = {"title": "VGGT-Gaussian: Feed-Forward 3D Gaussian Reconstruction"}
if _GR_MAJOR < 6:
    _blocks_kwargs["theme"] = _THEME

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.Markdown(_DESC)
    if not SCENE_NAMES:
        gr.Markdown(_NO_SCENE_TIP)

    gauss_cache = gr.State(None)  # caches filtered Gaussians for the scale slider

    with gr.Row():
        # ---------------- left: inputs ----------------
        with gr.Column(scale=5):
            gr.Markdown("### 1. Pick an example scene")
            scene_dd = gr.Dropdown(
                choices=SCENE_NAMES,
                value=(SCENE_NAMES[0] if SCENE_NAMES else None),
                label="Example scenes",
                interactive=True,
            )
            scene_preview = gr.Gallery(
                label="Input views of this scene",
                value=(_scene_gallery(SCENE_NAMES[0]) if SCENE_NAMES else []),
                columns=4, height=200, object_fit="cover", show_label=True,
            )

            run_btn = gr.Button("Run reconstruction", variant="primary", size="lg")

            _up_suffix = "(optional)" if ENABLE_UPLOAD else "(coming soon)"
            gr.Markdown(f"### 2. Or upload your own images {_up_suffix}")
            uploaded_state = gr.State([])
            upload_btn = gr.UploadButton(
                "Upload images (overrides the example)",
                file_count="multiple", file_types=["image"], variant="secondary",
                interactive=ENABLE_UPLOAD,
            )
            upload_status = gr.Markdown(
                "" if ENABLE_UPLOAD else "<sub>Upload is disabled for now.</sub>"
            )
            upload_preview = gr.Gallery(
                label="Uploaded images", columns=4, height=160,
                object_fit="cover", visible=ENABLE_UPLOAD,
            )
            clear_upload_btn = gr.Button(
                "Clear upload (use example again)", size="sm",
                interactive=ENABLE_UPLOAD,
            )

            gr.Markdown("### 3. Settings")
            preview_style = gr.Radio(
                choices=[PC_VIEW, SPLAT_VIEW],
                value=SPLAT_VIEW,
                label="Preview style",
            )
            _init_max_views = (_scene_num_images(SCENE_NAMES[0]) if SCENE_NAMES else 12) or 12
            num_views = gr.Slider(
                0, _init_max_views, value=0, step=1,
                label="Number of views to use (0 = all; more = denser but slower / more VRAM; 3-8 recommended)",
            )
            opacity_thr = gr.Slider(
                0.0, 0.99, value=0.1, step=0.01,
                label="opacity threshold (cull low-opacity 'fog' Gaussians; higher = cleaner)",
            )
            max_points = gr.Slider(
                10_000, 1_500_000, value=500_000, step=10_000,
                label="Max points / Gaussians (random subsample beyond this; also the preview point count)",
            )
            scale_mult = gr.Slider(
                0.2, 8.0, value=1.0, step=0.1,
                label="Splat scale x (enlarge each Gaussian; raise this if the splats look like separated dots)",
            )

        # ---------------- right: outputs ----------------
        with gr.Column(scale=7):
            gr.Markdown("### Result (left-drag = rotate, right-drag = pan, wheel = zoom)")
            viewer_html = gr.HTML(
                value="<div style='height:520px;display:flex;align-items:center;"
                      "justify-content:center;color:#888;border:1px dashed #ccc;"
                      "border-radius:8px;background:#fff'>Run a reconstruction to view here</div>"
            )
            stats_md = gr.Markdown("")
            with gr.Row():
                gs_file = gr.File(label="Download: Gaussian .ply (3DGS standard format)")
                pc_file = gr.File(label="Download: point cloud .ply (universal, MeshLab/CloudCompare)")
            input_gallery = gr.Gallery(
                label="Input views used this run", columns=4, height=160, object_fit="cover",
            )

    def _on_scene_change(scene_name):
        imgs = _scene_gallery(scene_name)
        n = _scene_num_images(scene_name) or 12
        # update gallery + rescale the views slider to this scene's image count
        return imgs, gr.update(maximum=n)

    scene_dd.change(_on_scene_change, inputs=scene_dd,
                    outputs=[scene_preview, num_views])

    def _on_upload(files):
        paths = _paths_from_upload(files)
        msg = (f"✅ {len(paths)} image(s) uploaded — used on the next run (overrides the example)."
               if paths else "")
        return paths, msg, paths

    upload_btn.upload(_on_upload, inputs=upload_btn,
                      outputs=[uploaded_state, upload_status, upload_preview])
    clear_upload_btn.click(lambda: ([], "", []),
                           outputs=[uploaded_state, upload_status, upload_preview])

    run_btn.click(
        run,
        inputs=[scene_dd, uploaded_state, num_views, preview_style, opacity_thr, max_points, scale_mult],
        outputs=[viewer_html, input_gallery, stats_md, gs_file, pc_file, gauss_cache],
    )

    # scale slider / preview style: re-render the cached result only (no re-inference)
    scale_mult.release(
        restyle,
        inputs=[gauss_cache, preview_style, scale_mult],
        outputs=viewer_html,
    )
    preview_style.change(
        restyle,
        inputs=[gauss_cache, preview_style, scale_mult],
        outputs=viewer_html,
    )

    gr.Markdown(
        "<sub>Point-cloud preview uses three.js TrackballControls (round soft points, free any-angle rotate/pan/zoom). "
        "Gaussian-splat preview uses gsplat.js for the continuous look (rotate/zoom). "
        "Both viewers render in your browser and need CDN access. The downloaded `.ply` files "
        "are full resolution and open in SuperSplat / MeshLab / CloudCompare.</sub>"
    )


if __name__ == "__main__":
    # Gradio only serves files from cwd / system temp / explicitly allowed paths.
    # Example scenes live under EXAMPLES_DIR (and OUTPUT_DIR for the .ply downloads),
    # which may be outside cwd -> whitelist them so the gallery / downloads work.
    # Use absolute paths; include REPO_ROOT (ancestor of everything) as a catch-all.
    _allowed = []
    for _d in (EXAMPLES_DIR, OUTPUT_DIR, REPO_ROOT, os.getcwd()):
        if not _d:
            continue
        _d = os.path.abspath(_d)
        if os.path.isdir(_d) and _d not in _allowed:
            _allowed.append(_d)
    print(f"[demo] allowed_paths = {_allowed}", flush=True)

    _launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=(os.environ.get("SHARE", "0") == "1"),
        show_error=True,
        allowed_paths=_allowed,
    )
    if _GR_MAJOR >= 6:
        _launch_kwargs["theme"] = _THEME
    try:
        demo.queue(max_size=16).launch(**_launch_kwargs)
    except TypeError:
        _launch_kwargs.pop("theme", None)
        demo.queue(max_size=16).launch(**_launch_kwargs)