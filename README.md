<div align="center">

# VGGT-Gaussian: Feed-Forward 3D Gaussian Reconstruction

**[Paper (IETCV, under review)](#)** | **[Project Page](#)** | **[arXiv](#)** | **[Video](#)**

</div>

<p align="center">
  <img src="demo_assets/pipeline.png" alt="VGGT-Gaussian Pipeline" width="90%">
</p>

---

## 📖 Introduction

**VGGT-Gaussian** builds on top of [VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded Transformer) and extends it with a feed-forward 3D Gaussian Splatting head, enabling direct regression of pixel-aligned 3D Gaussians from a set of input images **without per-scene optimization**. Given one or more RGB images, the model jointly predicts camera parameters, depth/point maps, and Gaussian primitive attributes (position, scale, rotation, opacity, color) in a single forward pass, which can then be rendered via differentiable Gaussian rasterization for novel view synthesis.

Key features:

- 🚀 **Feed-forward reconstruction** — no per-scene optimization required.
- 📷 **Sparse / unposed views supported** — leverages VGGT's pose-free geometric backbone.
- 🎯 **3D Gaussian Splatting output** — directly renderable, high-quality, real-time novel view synthesis.

## 🔧 Installation

### Option A: pip (local environment)

We recommend Python 3.10+ and CUDA 12.4

```bash
git clone https://github.com/gwj212/VGGT-G.git
cd VGGT-G

conda create -n vggt-gaussian python=3.10 -y
conda activate vggt-gaussian

pip install -r requirements.slim.txt
```

### Pretrained checkpoint


---

## ⚡ Quickstart / Demo

### 1. Run the interactive web demo

```bash
python app.py --ckpt ckpt.pth --port 7860
```

Then open `http://localhost:7860` in your browser. Sample images are provided under `demo_assets/`.

### 2. Standalone demo bundle

A self-contained demo package is also provided:

```bash
tar -xzvf vggt-gaussian-demo.tar.gz -C vggt_demo/
cd vggt_demo
python run_demo.py --input ./examples
```

> ✏️ Confirm the exact entry-point script name inside `vggt_demo/` and `vggt-gaussian-demo.tar.gz` — update `run_demo.py` above if it differs.

## 🏋️ Training

Training is launched via `train_nogt.py`. The `nogt` suffix indicates the **no-ground-truth-camera** training setting, where camera poses are not required as supervision (consistent with VGGT's pose-free design).

```bash
python train_nogt.py \
  --config configs/train_default.yaml \
  --data_root /path/to/dataset \
  --output_dir ./runs/vggt_gaussian_exp1 \
  --batch_size 4 \
  --num_gpus 8
```

Common arguments:

| Argument | Description | Default |
|---|---|---|
| `--config` | Path to training config (model/data/optim settings) | `configs/train_default.yaml` |
| `--data_root` | Root directory of the training dataset(s) | — |
| `--output_dir` | Directory for checkpoints/logs | `./runs/exp` |
| `--resume` | Path to a checkpoint to resume from | `None` |
| `--num_gpus` | Number of GPUs for DDP training | `1` |

Logs and intermediate checkpoints are written to `--output_dir`; training curves can be monitored via TensorBoard:

```bash
tensorboard --logdir ./runs/vggt_gaussian_exp1
```

> ✏️ Fill in the actual CLI flags/config schema of `train_nogt.py` (this table is a reasonable placeholder based on common conventions, not verified against your script).

---

## 📊 Evaluation

### Reconstruction / geometry evaluation

```bash
python eval.py \
  --ckpt ckpt.pth \
  --data_root /path/to/test_set \
  --output_dir ./eval_results
```

### Novel view synthesis (NVS) evaluation

```bash
python eval_nvs.py \
  --ckpt ckpt.pth \
  --data_root /path/to/test_set \
  --output_dir ./eval_nvs_results
```

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{vggtgaussian2026,
  title   = {VGGT-Gaussian: Feed-Forward 3D Gaussian Reconstruction},
  author  = {TODO: Author list},
  journal = {IET Computer Vision},
  year    = {2026},
  note    = {under review}
}
```

## 🙏 Acknowledgement

This project builds upon [VGGT](https://github.com/facebookresearch/vggt) and the 3D Gaussian Splatting rasterizer. We thank the authors of these works for open-sourcing their code.

## 📜 License

TODO: add license (e.g. MIT / Apache-2.0) consistent with the licenses of VGGT and any Gaussian Splatting rasterizer dependencies used.
