# vggt_demo/examples_util.py
# ============================================================
# Scan demo_assets/examples/<scene>/*.{jpg,png,...} into a list of example scenes.
# Each subdirectory = one multi-view scene.
# ============================================================

import os
import glob

EXAMPLES_DIR = os.environ.get("DEMO_EXAMPLES_DIR", "demo_assets/examples")
_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.webp")


def _images_in(directory):
    imgs = []
    for e in _EXTS:
        imgs.extend(glob.glob(os.path.join(directory, e)))
    imgs.sort()
    return imgs


def list_scenes(directory=None):
    """Return {scene_name: [image_paths...]} (sorted by name).
       If directory is None, use the module default (EXAMPLES_DIR)."""
    directory = directory or EXAMPLES_DIR
    scenes = {}
    if not os.path.isdir(directory):
        return scenes
    for name in sorted(os.listdir(directory)):
        sub = os.path.join(directory, name)
        if not os.path.isdir(sub):
            continue
        imgs = _images_in(sub)
        if imgs:
            scenes[name] = imgs
    return scenes


def scene_names():
    return list(list_scenes().keys())