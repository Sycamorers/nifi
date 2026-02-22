# Delivery Checklist

## Added Deliverables
- `scripts/fetch_real_dataset_example.py`
- `scripts/auto_select_best_demo_hparams.py`
- `configs/demo_realdata_known_good.yaml`
- Updated `scripts/run_demo.py` with dataset-aware CLI + quality gate
- Updated `scripts/export_demo_assets.py` for dataset/scene qualitative export
- `docs/assets/qualitative/mipnerf360/garden/view_000.png`
- `docs/assets/qualitative/mipnerf360/garden/view_001.png`
- `docs/assets/qualitative/mipnerf360/garden/view_002.png`
- `docs/REALDATA_DEMO_NOTES.md`
- Updated `README.md` with real-data qualitative results

## Reproduction Commands
1. Fetch real scene:
   - `python scripts/fetch_real_dataset_example.py --dataset mipnerf360 --scene garden --out data`
2. Select known-good hparams:
   - `python scripts/auto_select_best_demo_hparams.py --dataset mipnerf360 --scene garden --out_config configs/demo_realdata_known_good.yaml`
3. Run canonical demo:
   - `python scripts/run_demo.py --dataset mipnerf360 --scene garden --config configs/demo_realdata_known_good.yaml --views auto --seed 0`
4. Export README assets:
   - `python scripts/export_demo_assets.py --dataset mipnerf360 --scene garden --max_images 3`

## Verification Result
- Demo outputs generated at:
  - `outputs/demo/mipnerf360/garden/`
- Quality gate passed for showcased views:
  - LPIPS improved and sharpness non-decreasing on each selected view.
- README assets generated at:
  - `docs/assets/qualitative/mipnerf360/garden/`

## Core Files to Keep
- `scripts/fetch_real_dataset_example.py`
- `scripts/auto_select_best_demo_hparams.py`
- `scripts/run_demo.py`
- `scripts/export_demo_assets.py`
- `scripts/make_triptych.py`
- `configs/demo_realdata_known_good.yaml`
- `README.md`
- `docs/REALDATA_DEMO_NOTES.md`
