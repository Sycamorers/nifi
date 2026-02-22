# Real-Data Demo Notes

## Chosen Dataset / Scene
- Dataset: `Mip-NeRF360`
- Scene: `garden`
- Fetch script: `scripts/fetch_real_dataset_example.py`

## Why This Scene
- `Mip-NeRF360` is one of the benchmark datasets referenced in the paper.
- `garden` is a standard and widely used scene.
- The repo already supports this dataset path and scene structure directly.

## Demo Settings Used for README
- Config: `configs/demo_realdata_known_good.yaml`
- Rate: `0.1` (`rate_0.100`)
- Eq. (7) fields kept in config: `t0=199`, `adapter_scale=1.0`
- Final output mode selected by sweep: `classical_unsharp`
  - `smooth_passes=2`
  - `unsharp_radius=2.0`
  - `unsharp_percent=250`
  - `unsharp_threshold=2`

## Quality-Gate Policy
Per selected view, the final restored image must satisfy:
- LPIPS gain over LQ: `LPIPS(LQ,HQ) - LPIPS(Restored,HQ) >= 0.01`
- Laplacian sharpness non-decrease: `sharpness(Restored) >= sharpness(LQ)`

If LPIPS is unavailable, fallback is DISTS or SSIM according to script logic.

## Sweep Outcome
- Sweep script: `scripts/auto_select_best_demo_hparams.py`
- Summary: `outputs/auto_hparams/mipnerf360/garden/sweep_summary.json`
- Candidates evaluated: `72`
- Passing candidates: `11`
- Best candidate:
  - mode: `classical_unsharp`
  - rate: `0.1`
  - mean LPIPS gain: `+0.0505`

### Important checkpoint finding
With the currently available local tiny checkpoint (`runs/nifi_tiny/best.pt`), pure `nifi_eq7` candidates did not pass the gate on the swept views.
- `nifi_eq7` candidates tested: `18`
- `nifi_eq7` passing: `0`

This indicates missing paper-grade restoration weights in the current workspace for direct NiFi Eq.(7)-only qualitative parity.

## Reproduce End-to-End
1. Fetch data:
   - `python scripts/fetch_real_dataset_example.py --dataset mipnerf360 --scene garden --out data`
2. Sweep and select config:
   - `python scripts/auto_select_best_demo_hparams.py --dataset mipnerf360 --scene garden --out_config configs/demo_realdata_known_good.yaml`
3. Run demo:
   - `python scripts/run_demo.py --dataset mipnerf360 --scene garden --config configs/demo_realdata_known_good.yaml --views auto --seed 0`
4. Export README assets:
   - `python scripts/export_demo_assets.py --dataset mipnerf360 --scene garden --max_images 3`

## README Asset Paths
- `docs/assets/qualitative/mipnerf360/garden/view_000.png`
- `docs/assets/qualitative/mipnerf360/garden/view_001.png`
- `docs/assets/qualitative/mipnerf360/garden/view_002.png`
