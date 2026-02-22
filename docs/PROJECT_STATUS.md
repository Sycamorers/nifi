# Project Status (Deliverable Scope)

## Core Pipeline (Included)
- `scripts/fetch_real_dataset_example.py`: fetches one real benchmark scene (idempotent).
- `scripts/auto_select_best_demo_hparams.py`: sweeps restoration settings and emits quality-gated config.
- `scripts/run_demo.py`: canonical end-to-end qualitative demo entrypoint.
- `scripts/make_triptych.py`: labeled `[HQ | Compressed | Restored]` triptych builder.
- `scripts/export_demo_assets.py`: stable export into `docs/assets/qualitative/<dataset>/<scene>/`.
- `configs/demo_realdata_known_good.yaml`: known-good real-data demo config.

## Minimum Required Assets / Config
- Real scene data under `data/<dataset>/<scene>/` (e.g. `data/mipnerf360/garden/`).
- Restoration checkpoint (current default): `runs/nifi_tiny/best.pt`.
- Model config: `configs/default.yaml`.

## Excluded from Default Deliverable Path
- Large generated outputs (`outputs/`, `logs/`).
- Debug-only scripts quarantined under `tools/dev_debug/`.

## Known Assumptions
- Python dependencies from `requirements.txt` are installed.
- First run may download diffusion backbone weights from Hugging Face cache.
- Current local tiny checkpoint is not paper-grade; demo uses quality-gated selection to avoid blurry output.
