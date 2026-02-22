# Garden Training Report

Garden-only training/inference was run in conda env `nifi` on CUDA (`NVIDIA GeForce RTX 3090`).

## Commands Executed

```bash
conda run -n nifi python scripts/garden_env_gpu_check.py
conda run -n nifi python scripts/inspect_trainable_and_checkpoints.py
conda run -n nifi python scripts/train_garden_adapters.py --device cuda --max_steps 10000 --val_every 250 --save_every 1000
conda run -n nifi python scripts/auto_tune_garden.py --device cuda
conda run -n nifi python scripts/run_demo.py --dataset garden --device cuda --checkpoint checkpoints/garden/adapter_best.pt --config configs/garden_known_good.yaml
conda run -n nifi python scripts/quality_gate.py --demo_dir outputs/demo/garden/garden --garden_mode
```

## GPU Proof

- `logs/garden_env_gpu_check.txt`: `cuda_available=true`, device `cuda:0`, GPU `NVIDIA GeForce RTX 3090`.
- `logs/garden_train_10k.log`: startup block prints torch/CUDA/device and ongoing GPU memory/throughput logs.
- `logs/garden_train_run.md`: run summary with device and wall-clock duration.

## 10k Training Run Summary

- Run status: `ok` (`runs/garden_adapters/train_summary.json`)
- Total optimizer steps: `10000`
- Wall-clock: `4741.82 s` (~79.0 min)
- Mean throughput: `2.2238 steps/s` (`runs/garden_adapters/train_log.csv`)
- Mean GPU alloc: `3321.56 MB` (max alloc `3338.54 MB`, peak `13145.81 MB`)
- Loss trend: `0.55497 -> 0.45381`
- Debug+Adjust trigger after 10k: **not required** (`quality_gate_passed_during_training=true`)

Artifacts:
- Per-step curve CSV: `runs/garden_adapters/train_log.csv`
- Validation metrics curve CSV: `outputs/garden/metrics.csv`
- Validation triptychs every 250 steps: `outputs/garden/val/step_*/view_*/triptych.png`
- Periodic checkpoints: `checkpoints/garden/adapter_step*.pt`
- Periodic phi-minus exports: `checkpoints/garden/phi_minus_step*.safetensors`

## Quality Gate During Training

Gate criteria:
- `mean_abs_diff(restored, compressed) >= 0.01`
- `LPIPS(compressed, HQ) - LPIPS(restored, HQ) >= 0.02` (or SSIM fallback)
- `sharpness(restored) >= sharpness(compressed)`

Observed:
- 40 validation points (`outputs/garden/metrics.csv`)
- Gate passed during training: `true`
- `fixed_mean_lpips_gain`: min `0.01387`, max `0.02570`
- `fixed_mean_sharpness_gain`: min `0.0003829` (always non-negative)
- `fixed_mean_abs_diff`: min `0.06701` (always above threshold)

## Final Demo + Gate (Exported Result)

Final demo was generated with:
- Checkpoint: `checkpoints/garden/adapter_best.pt` (step `9750`, size `3.465 MB`)
- Config: `configs/garden_known_good.yaml`
- Manifest: `outputs/demo/garden/garden/manifest.json`

Final quality gate report:
- File: `outputs/demo/garden/garden/quality_gate_report.json`
- LPIPS mean: `0.72648 -> 0.70110` (gain `+0.02537`)
- Sharpness mean: `0.0009824 -> 0.0011534` (non-decreasing)
- Mean abs diff restored vs compressed: `0.07668`
- All views passed: `true`

## Final Qualitative Exports

- `docs/assets/qualitative/garden/garden/view_000.png`
- `docs/assets/qualitative/garden/garden/view_001.png`
- `docs/assets/qualitative/garden/garden/view_002.png`
- `docs/assets/qualitative/garden/garden/view_003.png`
