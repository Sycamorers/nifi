# NiFi Reproduction (Third-Party Learning Project)

Original paper:
- **Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models**
- arXiv: https://arxiv.org/abs/2602.04549

Disclaimer:
- This repository is an independent third-party implementation for learning and experimentation.

This repository is refactored so the code structure follows the paper workflow:
- Artifact Synthesis
- Artifact Restoration
- Restoration Distribution Matching
- Perceptual Matching
- Benchmark Evaluation

## 1. Environment Setup

### Python version
- `Python 3.11`

### Conda setup
```bash
conda env create -f environment.yml
conda activate nifi
```

### Pip setup (alternative)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Repository Structure

```text
configs/
  default.yaml

nifi/
  artifact_synthesis/
  artifact_restoration/
  restoration_distribution_matching/
  perceptual_matching/
  benchmark/

scripts/
  download_data.py
  download_benchmark_data.py
  build_3dgs_and_compress.py
  render_pairs.py
  prepare_benchmark_pairs.py
  train_nifi.py
  eval_nifi.py
  eval_benchmark_nifi.py
  verify_paper_implementation.py

docs/
  workflow_module_mapping.md
  equation_to_code_mapping.md
  implementation_verification.md
```

## 3. Dataset Preparation

## 3.1 Training data (DL3DV scenes)
The paper trains on DL3DV scenes. If you already have a local subset:
```bash
python scripts/download_data.py \
  --dataset dl3dv \
  --out data/ \
  --dl3dv_source /path/to/local/dl3dv_subset
```

## 3.2 Benchmark datasets (Sec. 4.2)
Download paper evaluation datasets:
```bash
python scripts/download_benchmark_data.py \
  --dataset all \
  --out data/benchmarks
```

Optional Mip-NeRF360 subset:
```bash
python scripts/download_benchmark_data.py \
  --dataset mipnerf360 \
  --out data/benchmarks \
  --mip_scenes garden bicycle kitchen
```

## 4. Preprocessing Pipeline

## 4.1 Artifact synthesis (clean + degraded renders)
```bash
python scripts/build_3dgs_and_compress.py \
  --scene data/mipnerf360/garden \
  --rates 0.1 0.5 1.0 \
  --out artifacts/garden/
```

## 4.2 Build pair layout for train/eval
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

Output layout:
```text
pairs/<scene>/rate_<lambda>/<split>/clean/*.png
pairs/<scene>/rate_<lambda>/<split>/degraded/*.png
```

## 4.3 Benchmark pair preprocessing
Proxy compression mode:
```bash
python scripts/prepare_benchmark_pairs.py \
  --dataset mipnerf360 \
  --dataset_root data/benchmarks/mipnerf360 \
  --out pairs/benchmarks \
  --rates 0.1 0.5 1.0 \
  --compression_method proxy
```

Precomputed HAC++ mode:
```bash
python scripts/prepare_benchmark_pairs.py \
  --dataset tanks_temples \
  --dataset_root data/benchmarks/tanks_temples \
  --out pairs/benchmarks \
  --rates 0.1 0.5 1.0 \
  --compression_method precomputed \
  --compressed_root /path/to/hacpp/renders
```

## 5. Training

Exact command:
```bash
python scripts/train_nifi.py \
  --config configs/default.yaml \
  --data_root pairs/ \
  --exp runs/nifi_train
```

### Paper-matched hyperparameters in `configs/default.yaml`
- `model.guidance_scale: 7.5`
- `model.lora_rank: 64`
- `diffusion.t0: 199`
- `train.batch_size: 4`
- `train.max_steps: 60000`
- `train.lr_phi_minus: 5e-6`
- `train.lr_phi_plus: 1e-6`
- `train.weight_decay: 1e-4`
- `train.grad_clip: 1.0`
- `loss_weights.alpha: 0.7`
- `model.prompt_dropout: 0.1`

Training outputs:
- `runs/<exp>/latest.pt`
- `runs/<exp>/best.pt`
- `runs/<exp>/train_log.csv`
- `runs/<exp>/train_summary.json`

## 6. Evaluation

## 6.1 Pair-root evaluation
```bash
python scripts/eval_nifi.py \
  --ckpt runs/nifi_train/best.pt \
  --data_root pairs/ \
  --split test \
  --out runs/nifi_train/metrics.json
```

Expected output format (`metrics.json`):
- `metrics.aggregate.lpips_before`
- `metrics.aggregate.lpips_after`
- `metrics.aggregate.dists_before`
- `metrics.aggregate.dists_after`
- `records[]` with per-image metrics

## 6.2 Benchmark protocol evaluation (Sec. 4.2)
```bash
python scripts/eval_benchmark_nifi.py \
  --ckpt runs/nifi_train/best.pt \
  --data_root pairs/benchmarks \
  --split test \
  --out runs/nifi_train/benchmark_metrics.json
```

Expected output format (`benchmark_metrics.json`):
- `summary.aggregate`
- `summary.per_dataset`
- `summary.per_dataset_rate`
- `summary.per_scene`
- `paper_comparison` (delta vs paper Table 1 NiFi numbers)

## 7. Reproducing Paper Results

Paper reports LPIPS/DISTS for NiFi at three rates (`lambda in {0.1, 0.5, 1.0}`):

- Mip-NeRF360: `0.178/0.109`, `0.235/0.133`, `0.265/0.153`
- Tanks & Temples: `0.128/0.076`, `0.180/0.095`, `0.212/0.109`
- DeepBlending: `0.133/0.101`, `0.180/0.131`, `0.218/0.156`

### Important deviation notes
Exact paper-level matching can deviate if any of these differ:
- Paper uses SD3 backbone; default config uses a lightweight SD-compatible model for practicality.
- Paper uses HAC++ low-rate compressed renders; proxy compression is used unless precomputed HAC++ outputs are supplied.
- Paper uses Qwen2.5-VL prompt extraction; this repo supports prompt files but not full automatic prompt generation workflow.

<!--
## 8. Formula-to-Code and Verification Docs

- Workflow mapping: `docs/workflow_module_mapping.md`
- Equation mapping: `docs/equation_to_code_mapping.md`
- Verification report: `docs/implementation_verification.md`

Run equation verification:
```bash
python scripts/verify_paper_implementation.py
```

Run unit tests:
```bash
python -m pytest -q tests/test_paper_alignment.py
```

## 9. One-Command Minimal Smoke Path

```bash
python scripts/train_nifi.py \
  --config configs/default.yaml \
  --data_root pairs/ \
  --exp runs/nifi_smoke \
  --smoke_test
```
!-->
