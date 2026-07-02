
import numpy as np

# SH degree-0 coefficient: rgb = C0 * f_dc + 0.5  ->  f_dc = (rgb - 0.5) / C0
_SH_C0 = 0.28209479177387814
_EPS = 1e-6


def _as_f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def filter_and_subsample(
    xyz, color, opacity=None, scale=None, rot=None,
    opacity_threshold=0.0, max_points=0, seed=0,
):
    """Cull by opacity threshold + random subsample down to max_points.

    All arrays are filtered by the same indices; returns
    (xyz, color, opacity, scale, rot, kept_idx).
    opacity/scale/rot may be None (not always needed for point-cloud mode).
    """
    xyz = _as_f32(xyz)
    color = _as_f32(color)
    n = xyz.shape[0]
    keep = np.ones(n, dtype=bool)

    if opacity is not None and opacity_threshold > 0.0:
        a = _as_f32(opacity).reshape(-1)
        keep &= (a >= float(opacity_threshold))

    idx = np.nonzero(keep)[0]

    if max_points and idx.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))

    def take(arr):
        return None if arr is None else _as_f32(arr)[idx]

    return (xyz[idx], color[idx],
            take(opacity), take(scale), take(rot), idx)


def _write_binary_ply(path, header_props, columns):
    """Generic binary little-endian ply writer.

    header_props: list[str]  vertex-property lines like ["property float x", ...]
    columns:      list[np.ndarray]  each shape (N,), in the same order as header_props
    """
    n = columns[0].shape[0]
    # Assemble (N, K) float32 column by column; force '<f4' for little-endian
    mat = np.empty((n, len(columns)), dtype='<f4')
    for k, col in enumerate(columns):
        mat[:, k] = col.astype('<f4', copy=False)

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(p + "\n" for p in header_props)
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(mat.tobytes(order="C"))
    return path


def write_3dgs_ply(path, xyz, color, opacity, scale, rot,
                   point_cloud_look=False, pc_scale=0.004, scale_mult=1.0):
    """Write a standard 3D Gaussian Splatting .ply.

    Args:
        xyz:     (N,3)  world coordinates
        color:   (N,3)  [0,1] RGB
        opacity: (N,1) or (N,)  post-sigmoid alpha in [0,1]
        scale:   (N,3)  post-exp real scale (world units)
        rot:     (N,4)  normalized quaternion, order (w,x,y,z)
        point_cloud_look: when True, flatten scale to a small uniform value +
                          opacity=1 so gsplat.js renders it as "dots" (point-cloud preview).
        pc_scale: the uniform scale used when point_cloud_look (world units).
    """
    xyz = _as_f32(xyz)
    color = _as_f32(color).clip(0.0, 1.0)
    n = xyz.shape[0]

    opacity = _as_f32(opacity).reshape(n)
    scale = _as_f32(scale).reshape(n, 3)
    rot = _as_f32(rot).reshape(n, 4)

    if scale_mult != 1.0:
        scale = scale * float(scale_mult)

    if point_cloud_look:
        opacity = np.ones(n, dtype=np.float32)
        scale = np.full((n, 3), float(pc_scale), dtype=np.float32)

    # pre-activation encoding
    f_dc = (color - 0.5) / _SH_C0                       # (N,3)
    a = np.clip(opacity, _EPS, 1.0 - _EPS)
    opacity_raw = np.log(a / (1.0 - a))                 # logit
    scale_raw = np.log(np.clip(scale, _EPS, None))      # log
    # normalize quaternion (safety), keep order (w,x,y,z)
    rn = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = rot / np.clip(rn, _EPS, None)

    normals = np.zeros((n, 3), dtype=np.float32)

    header_props = [
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
    ]
    columns = [
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        opacity_raw,
        scale_raw[:, 0], scale_raw[:, 1], scale_raw[:, 2],
        rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3],
    ]
    return _write_binary_ply(path, header_props, columns)


def write_pointcloud_ply(path, xyz, color):
    """Write a plain colored point cloud .ply (x,y,z float + red,green,blue uchar).

    ascii? No -- binary is smaller and faster. rgb is uchar here, handled separately.
    """
    xyz = _as_f32(xyz)
    color = _as_f32(color).clip(0.0, 1.0)
    n = xyz.shape[0]
    rgb = (color * 255.0 + 0.5).astype(np.uint8)

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    # structured dtype: 3 float + 3 uchar
    vtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.empty(n, dtype=vtype)
    arr["x"] = xyz[:, 0]; arr["y"] = xyz[:, 1]; arr["z"] = xyz[:, 2]
    arr["red"] = rgb[:, 0]; arr["green"] = rgb[:, 1]; arr["blue"] = rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes(order="C"))
    return path


# ---------------------------------------------------------------
# Mock Gaussian generator -- when torch/weights are unavailable, produce fake
# data with the SAME shape so export + frontend preview run end-to-end
# (used during frontend validation). Makes a colored spherical shell that
# looks like a 3D object.
# ---------------------------------------------------------------

def make_mock_gaussians(n=60000, seed=0):
    """Return (xyz, color, opacity, scale, rot), same shapes as the real merged output."""
    rng = np.random.default_rng(seed)

    # sphere + slight thickness for a "planet" feel
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 1.0 + rng.normal(0, 0.02, n)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.cos(phi)
    z = r * np.sin(phi) * np.sin(theta)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    # color varies with lat/long (looks nice + shows orientation)
    color = np.stack([
        0.5 + 0.5 * np.sin(theta),
        0.5 + 0.5 * np.cos(phi),
        0.5 + 0.5 * np.sin(phi) * np.cos(theta),
    ], axis=1).astype(np.float32).clip(0, 1)

    # mostly "surface" (high opacity), some "fog" (low opacity); mimics post-training distribution
    opacity = rng.uniform(0.7, 1.0, n).astype(np.float32)
    fog = rng.random(n) < 0.15
    opacity[fog] = rng.uniform(0.0, 0.1, fog.sum()).astype(np.float32)
    opacity = opacity[:, None]

    scale = np.full((n, 3), 0.01, dtype=np.float32) * rng.uniform(0.6, 1.6, (n, 1)).astype(np.float32)

    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # identity quaternion (w,x,y,z)

    return xyz, color, opacity, scale, rot


def splat_buffer(xyz, color, opacity, scale, rot,
                 point_cloud_look=False, pc_scale=0.01, scale_mult=1.0):
    """Build an antimatter15 '.splat' binary buffer (32 bytes per splat) for gsplat.js.

    Per-splat layout: position(3*f32) scale(3*f32) rgba(4*u8) rot(4*u8).
    Inputs are already activated: color/opacity in [0,1], scale linear,
    rot = (w,x,y,z) normalized. Returns raw bytes.
    """
    xyz = _as_f32(xyz)
    n = xyz.shape[0]
    color = _as_f32(color).clip(0.0, 1.0)
    buf = np.zeros((n, 32), dtype=np.uint8)

    buf[:, 0:12] = (np.ascontiguousarray(xyz, dtype="<f4")
                    .reshape(n, 3).view(np.uint8).reshape(n, 12))

    if point_cloud_look:
        sc = np.full((n, 3), float(pc_scale), dtype="<f4")
        a = np.ones(n, dtype=np.float32)
    else:
        sc = _as_f32(scale).reshape(n, 3).astype("<f4")
        if scale_mult != 1.0:
            sc = sc * np.float32(scale_mult)
        a = _as_f32(opacity).reshape(-1).clip(0.0, 1.0)
    buf[:, 12:24] = (np.ascontiguousarray(sc, dtype="<f4")
                     .reshape(n, 3).view(np.uint8).reshape(n, 12))

    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, 0:3] = (color * 255.0 + 0.5).astype(np.uint8)
    rgba[:, 3] = (a * 255.0 + 0.5).astype(np.uint8)
    buf[:, 24:28] = rgba

    q = _as_f32(rot).reshape(n, 4)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)
    buf[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return buf.tobytes()