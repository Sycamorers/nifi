# NiFi Repro-Lite (PyTorch)

Minimal end-to-end reproduction of **"Nix and Fix: Targeting 1000× Compression of 3D Gaussian Splatting with Diffusion Models"** with:

- artifact synthesis from clean views to degraded views (compression proxy + optional HAC++ hook)
- one-step latent diffusion restoration at intermediate timestep `t0`
- two LoRA-style adapters on a frozen latent diffusion backbone:
  - `phi_minus`: restoration adapter
  - `phi_plus`: critic adapter for distribution matching distillation
- alternating optimization (`phi_minus` then `phi_plus`)
- perceptual metrics: **LPIPS + DISTS**
- smoke-test mode (1 scene, 10 images, 100 steps)

This is a **repro-lite** implementation. It preserves NiFi training mechanics while using a practical diffusion substitution and a lightweight compression pipeline by default.

## 1. Repository Layout

```
configs/default.yaml
scripts/
  download_data.py
  build_3dgs_and_compress.py
  render_pairs.py
  train_nifi.py
  eval_nifi.py
nifi/
  data/
  gs/
  diffusion/
  losses/
  metrics/
  utils/
```

## 2. Install

```bash
conda env create -f environment.yml
conda activate nifi
```

If you are running CPU-only or macOS, remove `pytorch-cuda=12.1` from `environment.yml` before creating the environment.

For existing environments, update in-place:

```bash
conda env update -f environment.yml --prune
```

## 3. Design Choices (Important)

1. **Diffusion backbone substitution**
- Paper uses SD3 backbone.
- This repro defaults to `hf-internal-testing/tiny-stable-diffusion-torch` for fast verification.
- For closer fidelity, set `model.pretrained_model_name_or_path` in `configs/default.yaml` to SD1.5/2.1-compatible weights.

2. **3DGS compression backend**
- Default: `proxy` artifact synthesizer (prune/quantization-like degradations via downsampling, quantization, JPEG, blur).
- Optional: HAC++ wrapper hooks provided in `scripts/build_3dgs_and_compress.py` (`--method hacpp`).
- HAC++ upstream: `https://github.com/YihangChen-ee/HAC-plus`

3. **Prompt conditioning**
- Repro-lite uses empty prompts by default (unconditional path), but still keeps CFG and prompt dropout logic (`prompt_dropout` in config).

4. **NiFi mechanics implemented**
- Forward diffusion: `x_t = (1 - sigma_t) x + sigma_t eps`
- One-step restoration: `x_hat = x_tilde_t0 - sigma_t0 * eps_{theta,phi-}(x_tilde_t0, t0)`
- Distillation surrogate from score difference: `(s_restore - s_real)`
- Alternating updates:
  - update `phi_minus`: KL surrogate + GT guidance + L2 + LPIPS + DISTS
  - update `phi_plus`: diffusion noise prediction MSE on restored latents
- Default `t0=199`, LoRA rank `64`, `alpha=0.7` in config.

## 4. Data Download

### Mip-NeRF360 (real dataset)

Default downloads one scene (`garden`) from official GCS-hosted Mip-NeRF360 files.

```bash
python scripts/download_data.py --dataset mipnerf360 --out data/
```

Useful options:

```bash
# multiple scenes
python scripts/download_data.py --dataset mipnerf360 --out data/ --scenes garden bicycle

# create an images_tiny folder with first N images for quick debugging
python scripts/download_data.py --dataset mipnerf360 --out data/ --scenes garden --max_images 10
```

Notes:
- Scene zips are large (GB scale).
- `download_data.py` prepares `data/mipnerf360/<scene>/...`.

### DL3DV support

Manual subset copy is supported:

```bash
python scripts/download_data.py --dataset dl3dv --out data/ --dl3dv_source /path/to/local/dl3dv_subset
```

## 5. Artifact Synthesis + Pair Rendering

### Build 3DGS/compression artifacts

```bash
python scripts/build_3dgs_and_compress.py \
  --scene data/mipnerf360/garden \
  --rates 0.1 0.5 1.0 \
  --out artifacts/garden/
```

Optional tiny artifact generation:

```bash
python scripts/build_3dgs_and_compress.py \
  --scene data/mipnerf360/garden \
  --rates 0.1 0.5 1.0 \
  --max_images 10 \
  --out artifacts/garden/
```

### Render aligned clean/degraded pairs

```bash
python scripts/render_pairs.py \
  --scene artifacts/garden/ \
  --split train \
  --out pairs/garden/train/

python scripts/render_pairs.py \
  --scene artifacts/garden/ \
  --split test \
  --out pairs/garden/
```

`render_pairs.py` normalizes output roots, so both `pairs/<scene>/` and `pairs/<scene>/train/` forms are accepted.

This produces:

```
pairs/garden/rate_0.100/train/{clean,degraded}
pairs/garden/rate_0.500/train/{clean,degraded}
pairs/garden/rate_1.000/train/{clean,degraded}
...
```

## 6. Train

```bash
python scripts/train_nifi.py \
  --config configs/default.yaml \
  --data_root pairs/ \
  --exp runs/nifi_tiny
```

### Single GPU (RTX 3090, 24GB) runtime

- `configs/default.yaml` now includes a `runtime` block for GPU behavior:
  - `device: cuda`, `device_id: 0`
  - `mixed_precision: fp16`
  - `allow_tf32: true`, `cudnn_benchmark: true`
  - pinned/non-blocking dataloader transfer enabled
- To force one GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_nifi.py \
  --config configs/default.yaml \
  --data_root pairs/ \
  --exp runs/nifi_3090
```

### Smoke test (requested minimal verification path)

```bash
python scripts/train_nifi.py \
  --config configs/default.yaml \
  --data_root pairs/ \
  --exp runs/nifi_smoke \
  --smoke_test
```

Smoke mode uses config defaults of roughly:
- 10 images
- 100 steps
- fast eval subset

Training outputs:
- `runs/<exp>/latest.pt`
- `runs/<exp>/best.pt`
- `runs/<exp>/train_log.csv`
- `runs/<exp>/train_summary.json`

## 7. Evaluate (LPIPS + DISTS)

```bash
python scripts/eval_nifi.py \
  --ckpt runs/nifi_tiny/best.pt \
  --data_root pairs/ \
  --split test \
  --out runs/nifi_tiny/metrics.json
```

Output files:
- JSON: per-image, per-scene, aggregate metrics
- CSV: flat per-image table with LPIPS/DISTS before and after restoration

## 8. Expected Commands (from task)

```bash
python scripts/download_data.py --dataset mipnerf360 --out data/
python scripts/build_3dgs_and_compress.py --scene data/mipnerf360/<scene> --rates 0.1 0.5 1.0 --out artifacts/<scene>/
python scripts/render_pairs.py --scene artifacts/<scene>/ --split train --out pairs/<scene>/train/
python scripts/train_nifi.py --config configs/default.yaml --data_root pairs/ --exp runs/nifi_tiny
python scripts/eval_nifi.py --ckpt runs/nifi_tiny/best.pt --data_root pairs/ --split test --out runs/nifi_tiny/metrics.json
```

## 8.1 Self-test (Auto Diagnostics + Auto Fallback)

Run one command to execute build -> pair rendering -> metrics with hang detection, diagnostics, and retries:

```bash
python scripts/selftest_pipeline.py \
  --scene data/mipnerf360/garden \
  --out artifacts/garden_selftest \
  --rates 0.5 \
  --smoke
```

What success looks like:
- Console ends with `PASS`
- `artifacts/garden_selftest/` contains:
  - `run_manifest.json` (stage timings + environment + artifacts)
  - `selftest_manifest.json` (step retries + final assertions)
  - `logs/*.log` (captured subprocess output)
  - at least one readable `.png`
  - `metrics.json` and `metrics.csv` with numeric metrics

If any step hangs (>60s no output) or times out, diagnostics are auto-dumped:
- process tree
- CPU/memory snapshot
- GPU utilization/VRAM snapshot
- recent output file writes
- best-effort Python stack traces via `SIGUSR1` + `faulthandler`

## 9. Robustness Features

- deterministic seeds (`nifi/utils/seed.py`)
- checkpoint save/resume (`nifi/utils/checkpoint.py`)
- fp16/bf16 autocast support
- optimized single-GPU runtime controls (`runtime` block in `configs/default.yaml`)
- gradient clipping
- NaN/Inf checks + tensor shape checks
- tiny overfit sanity guard (start/end loss trend)

## 10. Known Limitations

- Default compressor is a practical proxy, not full HAC++ unless externally configured.
- Default diffusion model is tiny for reproducibility; quality is lower than SD1.5/SD3-scale backbones.
- Mip-NeRF360 downloads are large.
