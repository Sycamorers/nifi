# NiFi Paper-to-Code Map (Re-audit)

This document maps key NiFi/NIX&FIX paper components (arXiv:2602.04549) to concrete repository code and runtime behavior.

## 1) What is the "compressed render" input?
- Paper intent:
  - `I~` is a low-rate 3DGS render produced after pruning/quantization/entropy coding.
- Repo implementation path:
  - Compression proxy: `nifi/gs/compressor.py` -> `Proxy3DGSCompressor.degrade_image`
  - Artifact synthesis wrapper: `nifi/artifact_synthesis/compression_simulation.py` -> `ProxyArtifactSynthesisCompressor.synthesize_view_artifact`
  - Scene-level generation: `scripts/build_3dgs_and_compress.py` -> `stage_render_degraded`
  - Demo fallback generation from real dataset frames: `scripts/run_demo.py` -> `_resolve_artifact_scene`, `_ensure_rate_dir`
- Current practical behavior:
  - For demo runs, HQ is sourced from real dataset images and LQ is generated via proxy compression unless a prepared artifact scene already exists.

## 2) Eq. (7) one-step restore with `t0`
- Paper equation:
  - `x^ = x~_{t0} - sigma_{t0} * eps_{theta,phi-}(x~_{t0}, t0)`
- Repo implementation:
  - Core Eq. (7): `nifi/artifact_restoration/model.py` -> `artifact_restoration_one_step_eq7`
  - Demo inference path: `scripts/run_demo.py` -> `DemoRestorer.restore_eq7`
  - Diagnostic proof path: `scripts/prove_restoration_is_applied.py`
- `t0` selection:
  - Demo config value: `configs/demo_realdata_known_good.yaml` -> `restoration.t0`
  - Timesteps routed to sigma schedule in `nifi/diffusion/schedule.py`

## 3) Backbone and trainable/frozen components
- Paper states:
  - Frozen diffusion backbone (SD3 in paper), trainable low-rank adapters `phi-` and `phi+`.
- Repo implementation:
  - Backbone + adapters: `nifi/diffusion/model.py` -> `FrozenLDMWithNiFiAdapters`
  - Frozen modules: text encoder, VAE, UNet (`requires_grad_(False)`)
  - Trainable adapters: `phi_minus`, `phi_plus` (`LowRankSpatialAdapter`)
- Config source:
  - Train/demo backbone is controlled by model config YAML (`model.pretrained_model_name_or_path`).
  - Current model-path source of truth: `configs/model_paths.yaml`.

## 4) Where LPIPS/L2/perceptual terms appear
- L2 + LPIPS (+ optional DISTS):
  - `nifi/losses/perceptual.py` -> `ReconstructionLossBundle.forward`
- Eq. (6) composition with distribution matching terms:
  - `nifi/restoration_distribution_matching/objectives.py` -> `phi_minus_objective_eq6`
- Training usage:
  - `scripts/train_nifi.py` -> inside main loop where `rec_losses(clean, restored)` and `phi_minus_objective_eq6(...)` are combined.

## 5) What must happen at inference
- Required steps:
  1. Load compatible model config + adapter checkpoint.
  2. Encode LQ image to latent (`E`).
  3. Forward project to `t0` (`q_sample`).
  4. Predict noise with `phi-`.
  5. Apply Eq. (7) one-step update.
  6. Decode latent (`D`) to restored RGB.
- Repo inference entry:
  - `scripts/run_demo.py` -> `main` + `DemoRestorer.restore_eq7`
- Validation/proof:
  - `scripts/prove_restoration_is_applied.py` logs latent stats, sigma/timestep, and output deltas.

## 6) Renderer entry for HQ / LQ / restoration
- HQ render/export entry:
  - `scripts/build_3dgs_and_compress.py` -> `stage_render_degraded` (clean view export)
  - Demo path also reads `artifacts/<scene>/clean` directly (`scripts/run_demo.py`)
- LQ compressed render entry:
  - `Proxy3DGSCompressor.degrade_image` called by:
    - `scripts/build_3dgs_and_compress.py`
    - `scripts/run_demo.py` (`_ensure_rate_dir`)
- Restoration inference entry:
  - `scripts/run_demo.py` -> `DemoRestorer.restore_eq7`
  - `scripts/eval_nifi.py` -> `artifact_restoration_one_step_eq7`

## 7) Checkpoint loading and expected paths
- Single source of truth:
  - `configs/model_paths.yaml`:
    - `restoration.model_config`
    - `restoration.adapter_checkpoint`
- Runtime loaders:
  - `scripts/run_demo.py` -> `_build_restorer(...)`
  - `scripts/verify_checkpoints.py` -> hard verification + compatibility checks
- Raw checkpoint reader:
  - `nifi/utils/checkpoint.py` -> `load_checkpoint`

## 8) Scheduler implementation and `t0` mapping
- Sigma schedule:
  - `nifi/diffusion/schedule.py` -> `SigmaSchedule`
  - `sigma(t)` from `registered_sigmas[t]`
  - `q_sample(x,t) = (1-sigma_t)x + sigma_t*noise`
- `t0` consumption:
  - `scripts/run_demo.py` passes `restoration.t0` into restore path
  - `scripts/prove_restoration_is_applied.py` prints selected timestep index and `sigma_t0`

## 9) Re-audit findings that explain ineffective restoration
- H1 (restore bypass): default demo previously allowed non-NiFi output selection (`classical_unsharp`), so learned restoration could be bypassed.
- H2 (checkpoint mismatch/missing): adapter checkpoints were tied to tiny test backbone shapes and not valid for real SD backbones.
- H3/H5 (latent/range/resolution): range conversion and pad/unpad paths exist and are now explicitly checked in diagnostics.
- H4 (scheduler/t0): `t0` is configurable; diagnostics now log exact timestep index and `sigma_t0`.
- H7/H8 context: demo uses proxy artifact synthesis from real dataset frames; no claim of reproducing full SD3/HAC++ paper training without official assets/weights.

