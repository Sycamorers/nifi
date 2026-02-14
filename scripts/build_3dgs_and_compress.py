#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.gs import CompressionConfig, HACPPWrapper, Proxy3DGSCompressor, list_scene_images
from nifi.utils.diagnostics import (
    collect_diagnostics,
    dump_stack_traces,
    enable_faulthandler_signal,
    write_diagnostics,
)
from nifi.utils.logging import dump_json, get_logger
from nifi.utils.seed import set_seed


STAGE_ORDER = [
    "validate_input",
    "setup_env",
    "gs_train_or_load",
    "compress",
    "render_degraded",
    "sanity_output",
]


class StageFailed(RuntimeError):
    pass


@dataclass
class StageProgress:
    done: int = 0
    total: int = 0
    last_substep: str = "init"
    last_progress_ts: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, done: Optional[int] = None, total: Optional[int] = None, substep: Optional[str] = None) -> None:
        with self._lock:
            if done is not None:
                self.done = int(done)
            if total is not None:
                self.total = int(total)
            if substep is not None:
                self.last_substep = str(substep)
            self.last_progress_ts = time.time()

    def mark_substep(self, substep: str) -> None:
        self.update(substep=substep)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "done": self.done,
                "total": self.total,
                "last_substep": self.last_substep,
                "last_progress_ts": self.last_progress_ts,
            }



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 3DGS artifacts and compress scenes with observability")
    p.add_argument("--scene", type=str, required=True, help="Path to dataset scene directory")
    p.add_argument("--rates", type=float, nargs="+", required=True, help="Compression rates")
    p.add_argument("--out", type=str, required=True, help="Artifact output directory")
    p.add_argument("--method", type=str, default="proxy", choices=["proxy", "hacpp"])

    p.add_argument("--holdout_every", type=int, default=8)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--render_max_views", type=int, default=None)
    p.add_argument("--render_short_side", type=int, default=None)

    p.add_argument("--timeout_stage_sec", type=int, default=1800)
    p.add_argument("--heartbeat_sec", type=int, default=10)
    p.add_argument("--idle_timeout_sec", type=int, default=60)

    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_compress", action="store_true")
    p.add_argument("--use_cpu", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Enable smoke defaults")

    p.add_argument("--jpeg_quality_min", type=int, default=8)
    p.add_argument("--jpeg_quality_max", type=int, default=55)
    p.add_argument("--downsample_min", type=int, default=2)
    p.add_argument("--downsample_max", type=int, default=8)

    p.add_argument("--hacpp_repo", type=str, default=None)
    p.add_argument("--hacpp_train_cmd", type=str, default="")
    p.add_argument("--hacpp_compress_cmd", type=str, default="")
    return p.parse_args()



def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if args.max_images is None:
        args.max_images = 10
    if args.max_steps is None:
        args.max_steps = 100
    if args.render_max_views is None:
        args.render_max_views = 4
    if args.render_short_side is None:
        args.render_short_side = 320



def get_git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
        commit = proc.stdout.strip()
        return commit if commit else None
    except Exception:
        return None



def configure_logger(debug: bool) -> logging.Logger:
    logger = get_logger("nifi.build")
    if debug:
        logger.setLevel(logging.DEBUG)
    return logger



def resize_short_side(img: Image.Image, short_side: Optional[int]) -> Image.Image:
    if short_side is None:
        return img
    short_side = int(short_side)
    if short_side <= 0:
        return img

    w, h = img.size
    cur_short = min(w, h)
    if cur_short == short_side:
        return img

    scale = float(short_side) / float(cur_short)
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)



def run_stage(
    name: str,
    fn: Callable[[argparse.Namespace, Dict[str, Any], StageProgress, logging.Logger], None],
    args: argparse.Namespace,
    state: Dict[str, Any],
    logger: logging.Logger,
    manifest: Dict[str, Any],
) -> None:
    logger.info("[stage:start] %s args={scene=%s,out=%s,method=%s}", name, args.scene, args.out, args.method)

    progress = StageProgress(last_substep="stage_start")
    stage_start = time.time()
    err_box: Dict[str, Any] = {}

    def _target() -> None:
        try:
            fn(args, state, progress, logger)
        except Exception as e:
            err_box["error"] = e
            err_box["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=_target, name=f"stage-{name}", daemon=True)
    thread.start()

    last_heartbeat = 0.0

    while thread.is_alive():
        now = time.time()
        snap = progress.snapshot()
        elapsed = now - stage_start

        if now - last_heartbeat >= float(args.heartbeat_sec):
            done = snap["done"]
            total = snap["total"]
            if total > 0:
                logger.info(
                    "[stage:heartbeat] %s elapsed=%.1fs progress=%d/%d last_substep=%s",
                    name,
                    elapsed,
                    done,
                    total,
                    snap["last_substep"],
                )
            else:
                logger.info(
                    "[stage:heartbeat] %s elapsed=%.1fs last_substep=%s",
                    name,
                    elapsed,
                    snap["last_substep"],
                )
            last_heartbeat = now

        idle = now - float(snap["last_progress_ts"])
        if idle > float(args.idle_timeout_sec):
            msg = f"Stage '{name}' produced no progress for {idle:.1f}s"
            logger.error(msg)
            diag = collect_diagnostics(name, state["out_dir"], snap["last_substep"], pid=os.getpid())
            diag["reason"] = "idle_timeout"
            diag["idle_seconds"] = idle
            dump_stack_traces(os.getpid())
            diag_path = state["out_dir"] / f"diagnostics_{name}_idle.json"
            write_diagnostics(diag_path, diag)
            manifest["stage_status"][name] = "failed_idle"
            manifest["diagnostics"].append(str(diag_path))
            raise StageFailed(msg)

        if elapsed > float(args.timeout_stage_sec):
            msg = f"Stage '{name}' exceeded timeout {args.timeout_stage_sec}s"
            logger.error(msg)
            diag = collect_diagnostics(name, state["out_dir"], snap["last_substep"], pid=os.getpid())
            diag["reason"] = "stage_timeout"
            diag["elapsed_seconds"] = elapsed
            dump_stack_traces(os.getpid())
            diag_path = state["out_dir"] / f"diagnostics_{name}_timeout.json"
            write_diagnostics(diag_path, diag)
            manifest["stage_status"][name] = "failed_timeout"
            manifest["diagnostics"].append(str(diag_path))
            raise StageFailed(msg)

        time.sleep(0.5)

    thread.join(timeout=0.1)

    if "error" in err_box:
        logger.error("[stage:error] %s failed:\n%s", name, err_box["traceback"])
        snap = progress.snapshot()
        diag = collect_diagnostics(name, state["out_dir"], snap["last_substep"], pid=os.getpid())
        diag["reason"] = "exception"
        diag["exception"] = str(err_box["error"])
        diag_path = state["out_dir"] / f"diagnostics_{name}_exception.json"
        write_diagnostics(diag_path, diag)
        manifest["stage_status"][name] = "failed_exception"
        manifest["diagnostics"].append(str(diag_path))
        raise StageFailed(f"Stage '{name}' failed: {err_box['error']}")

    elapsed = time.time() - stage_start
    manifest["stage_timings"][name] = elapsed
    manifest["stage_status"][name] = "ok"
    logger.info("[stage:end] %s elapsed=%.2fs", name, elapsed)



def stage_validate_input(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    scene_dir = Path(args.scene)
    out_dir = Path(args.out)

    progress.update(total=5, done=0, substep="check_scene_dir")
    if not scene_dir.exists() or not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene path not found or not a directory: {scene_dir}")
    progress.update(done=1, substep="check_out_dir")

    out_dir.mkdir(parents=True, exist_ok=True)
    test_write = out_dir / ".write_test"
    test_write.write_text("ok")
    test_write.unlink(missing_ok=True)

    progress.update(done=2, substep="scan_images")
    images = list_scene_images(scene_dir)
    if not images:
        raise RuntimeError(f"No images found under scene: {scene_dir}")

    selected = images
    if args.max_images is not None:
        selected = selected[: int(args.max_images)]
    if args.render_max_views is not None:
        selected = selected[: int(args.render_max_views)]

    if not selected:
        raise RuntimeError("Image selection is empty after max_images/render_max_views filters")

    progress.update(done=3, substep="camera_file_checks")
    cam_files = [
        scene_dir / "sparse" / "0" / "cameras.bin",
        scene_dir / "sparse" / "0" / "cameras.txt",
    ]
    img_files = [
        scene_dir / "sparse" / "0" / "images.bin",
        scene_dir / "sparse" / "0" / "images.txt",
    ]
    cam_exists = any(p.exists() for p in cam_files)
    img_exists = any(p.exists() for p in img_files)
    if not cam_exists or not img_exists:
        logger.warning(
            "COLMAP camera/image metadata missing or partial under %s (camera_ok=%s image_ok=%s). "
            "Proxy mode can proceed, but true 3DGS backends may fail.",
            scene_dir / "sparse" / "0",
            cam_exists,
            img_exists,
        )

    progress.update(done=4, substep="store_state")
    state["scene_dir"] = scene_dir
    state["out_dir"] = out_dir
    state["images_all"] = [str(p) for p in images]
    state["images_selected"] = [str(p) for p in selected]
    state["num_images_selected"] = len(selected)

    progress.update(done=5, substep="validate_input_complete")
    logger.info("validate_input: found %d images, selected %d", len(images), len(selected))



def stage_setup_env(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    progress.update(total=6, done=0, substep="seed")
    set_seed(int(args.seed), deterministic=False)

    progress.update(done=1, substep="thread_env")
    if args.smoke:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    progress.update(done=2, substep="torch_threads")
    try:
        if "OMP_NUM_THREADS" in os.environ:
            torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))
    except Exception:
        pass

    progress.update(done=3, substep="device_selection")
    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cuda_available = torch.cuda.is_available() and not args.use_cpu

    progress.update(done=4, substep="gpu_info")
    gpu_info = None
    if cuda_available:
        try:
            dev = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(dev)
            gpu_info = {
                "name": props.name,
                "vram_gb": props.total_memory / (1024 ** 3),
            }
        except Exception:
            gpu_info = {"name": "unknown", "vram_gb": None}

    progress.update(done=5, substep="versions")
    env_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": gpu_info,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    state["env_info"] = env_info

    progress.update(done=6, substep="setup_env_complete")
    logger.info("setup_env: %s", json.dumps(env_info, indent=2))



def stage_gs_train_or_load(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    out_dir = Path(state["out_dir"])
    gs_dir = out_dir / "gs"
    gs_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        progress.update(total=1, done=1, substep="skip_train")
        state["gs_checkpoint"] = None
        logger.info("gs_train_or_load: skipped by --skip_train")
        return

    if args.method == "proxy":
        max_steps = int(args.max_steps if args.max_steps is not None else 200)
        max_steps = max(1, max_steps)
        progress.update(total=max_steps, done=0, substep="proxy_train_start")

        # Proxy mode does not train real 3DGS, but emits progress and a tiny checkpoint marker.
        interval = max(1, max_steps // 20)
        for step in range(1, max_steps + 1):
            if step == 1 or step == max_steps or (step % interval == 0):
                progress.update(done=step, total=max_steps, substep=f"proxy_train_step_{step}")
            # Keep this stage light; no heavy compute in repro-lite.

        ckpt = gs_dir / "proxy_gs_checkpoint.json"
        ckpt.write_text(json.dumps({"backend": "proxy", "steps": max_steps, "seed": int(args.seed)}, indent=2))
        state["gs_checkpoint"] = str(ckpt)
        progress.update(done=max_steps, total=max_steps, substep="proxy_train_complete")
        logger.info("gs_train_or_load: wrote %s", ckpt)
        return

    # HAC++ path validation only; actual train/compress is run in compress stage.
    progress.update(total=1, done=0, substep="validate_hacpp")
    if not args.hacpp_repo:
        raise ValueError("--hacpp_repo is required when --method hacpp")
    if not args.hacpp_train_cmd or not args.hacpp_compress_cmd:
        raise ValueError("--hacpp_train_cmd and --hacpp_compress_cmd are required when --method hacpp")
    progress.update(total=1, done=1, substep="validate_hacpp_complete")



def stage_compress(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    out_dir = Path(state["out_dir"])
    rates = [float(r) for r in args.rates]

    if args.skip_compress:
        progress.update(total=1, done=1, substep="skip_compress")
        state["compressor"] = None
        state["compression_mode"] = "bypass"
        logger.info("compress: skipped by --skip_compress")
        return

    if args.method == "proxy":
        cfg = CompressionConfig(
            jpeg_quality_min=int(args.jpeg_quality_min),
            jpeg_quality_max=int(args.jpeg_quality_max),
            downsample_min=int(args.downsample_min),
            downsample_max=int(args.downsample_max),
        )
        compressor = Proxy3DGSCompressor(cfg)

        progress.update(total=len(rates), done=0, substep="create_rate_dirs")
        for i, rate in enumerate(rates, start=1):
            rate_name = f"rate_{float(rate):.3f}"
            (out_dir / rate_name).mkdir(parents=True, exist_ok=True)
            progress.update(done=i, total=len(rates), substep=f"prepare_{rate_name}")

        state["compressor"] = compressor
        state["compression_mode"] = "proxy"
        state["compression_cfg"] = {
            "jpeg_quality_min": int(args.jpeg_quality_min),
            "jpeg_quality_max": int(args.jpeg_quality_max),
            "downsample_min": int(args.downsample_min),
            "downsample_max": int(args.downsample_max),
        }
        return

    # HAC++ path: execute external train+compress wrapper.
    progress.update(total=1, done=0, substep="hacpp_wrapper_run")
    wrapper = HACPPWrapper(Path(args.hacpp_repo))
    wrapper.run(
        scene_dir=Path(args.scene),
        out_dir=Path(args.out),
        rates=rates,
        train_cmd_template=args.hacpp_train_cmd,
        compress_cmd_template=args.hacpp_compress_cmd,
    )
    state["compression_mode"] = "hacpp"
    state["compressor"] = None
    progress.update(total=1, done=1, substep="hacpp_wrapper_complete")



def stage_render_degraded(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    out_dir = Path(state["out_dir"])
    rates = [float(r) for r in args.rates]

    if args.method == "hacpp":
        # Assume HAC++ path rendered/produced outputs externally.
        progress.update(total=1, done=1, substep="hacpp_render_external")
        logger.info("render_degraded: skipped internal rendering for HAC++ mode")
        return

    selected = [Path(p) for p in state["images_selected"]]
    clean_dir = out_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    total = len(selected) + (len(selected) * len(rates))
    done = 0
    progress.update(total=total, done=0, substep="render_clean_start")

    # Stage 1: produce aligned clean views.
    clean_names: List[str] = []
    for i, src in enumerate(selected):
        name = f"{i:05d}.png"
        clean_names.append(name)

        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb = resize_short_side(rgb, args.render_short_side)
            rgb.save(clean_dir / name)

        done += 1
        progress.update(done=done, total=total, substep=f"clean_view_{i+1}/{len(selected)}")

    # Stage 2: produce degraded views per compression rate.
    compressor: Optional[Proxy3DGSCompressor] = state.get("compressor", None)
    skip_compress = bool(args.skip_compress)

    for r_idx, rate in enumerate(rates, start=1):
        rate_name = f"rate_{float(rate):.3f}"
        rate_dir = out_dir / rate_name
        rate_dir.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(clean_names):
            with Image.open(clean_dir / name) as clean_img:
                clean_rgb = clean_img.convert("RGB")
                if skip_compress or compressor is None:
                    degraded = clean_rgb.copy()
                else:
                    degraded = compressor.degrade_image(clean_rgb, float(rate))
                degraded.save(rate_dir / name)

            done += 1
            progress.update(
                done=done,
                total=total,
                substep=f"render_{rate_name}_{i+1}/{len(clean_names)}",
            )

    # Stage 3: write metadata + split.
    splits = {"train": [], "test": []}
    for i, name in enumerate(clean_names):
        if args.holdout_every > 0 and i % int(args.holdout_every) == 0:
            splits["test"].append(name)
        else:
            splits["train"].append(name)

    meta = {
        "scene_dir": str(state["scene_dir"]),
        "num_images": len(clean_names),
        "rates": rates,
        "splits": splits,
        "compression": {
            "backend": "proxy" if not skip_compress else "bypass",
            "jpeg_quality_min": int(args.jpeg_quality_min),
            "jpeg_quality_max": int(args.jpeg_quality_max),
            "downsample_min": int(args.downsample_min),
            "downsample_max": int(args.downsample_max),
        },
        "render": {
            "render_short_side": int(args.render_short_side) if args.render_short_side is not None else None,
            "render_max_views": int(args.render_max_views) if args.render_max_views is not None else None,
        },
    }

    meta_path = out_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    state["metadata"] = meta
    state["metadata_path"] = str(meta_path)
    state["clean_dir"] = str(clean_dir)



def stage_sanity_output(args: argparse.Namespace, state: Dict[str, Any], progress: StageProgress, logger: logging.Logger) -> None:
    out_dir = Path(state["out_dir"])
    progress.update(total=5, done=0, substep="check_metadata")

    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    progress.update(done=1, substep="check_clean_png")
    clean_pngs = sorted((out_dir / "clean").glob("*.png"))
    if not clean_pngs:
        raise RuntimeError(f"No clean PNG files found in {out_dir / 'clean'}")

    progress.update(done=2, substep="check_degraded_png")
    degraded_pngs = sorted(out_dir.glob("rate_*/*.png"))
    if not degraded_pngs:
        raise RuntimeError(f"No degraded PNG files found under {out_dir}")

    progress.update(done=3, substep="verify_png_readability")
    sample = degraded_pngs[0]
    if sample.stat().st_size <= 0:
        raise RuntimeError(f"Sample PNG has zero size: {sample}")

    with Image.open(sample) as img:
        img.verify()

    progress.update(done=4, substep="store_artifacts")
    state["artifact_paths"] = {
        "metadata": str(meta_path),
        "clean_dir": str(out_dir / "clean"),
        "sample_png": str(sample),
    }

    progress.update(done=5, substep="sanity_output_complete")
    logger.info("sanity_output: sample_png=%s size=%d", sample, sample.stat().st_size)



def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_order": list(STAGE_ORDER),
        "stage_timings": {},
        "stage_status": {},
        "diagnostics": [],
        "artifacts": {},
    }



def main() -> None:
    enable_faulthandler_signal()

    args = parse_args()
    apply_smoke_defaults(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(debug=bool(args.debug))
    manifest = build_manifest(args)

    state: Dict[str, Any] = {
        "out_dir": out_dir,
    }

    stage_map = {
        "validate_input": stage_validate_input,
        "setup_env": stage_setup_env,
        "gs_train_or_load": stage_gs_train_or_load,
        "compress": stage_compress,
        "render_degraded": stage_render_degraded,
        "sanity_output": stage_sanity_output,
    }

    start = time.time()
    try:
        for stage_name in STAGE_ORDER:
            run_stage(stage_name, stage_map[stage_name], args, state, logger, manifest)

        manifest["status"] = "success"
        manifest["artifacts"] = state.get("artifact_paths", {})
        manifest["elapsed_sec"] = time.time() - start
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        manifest_path = out_dir / "run_manifest.json"
        dump_json(manifest_path, manifest)

        # Keep backward compatibility with previous metadata-only stdout behavior.
        meta = state.get("metadata", None)
        if meta is not None:
            print(json.dumps(meta, indent=2))
        logger.info("Build completed successfully. run_manifest=%s", manifest_path)

    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        manifest["elapsed_sec"] = time.time() - start
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        fail_diag = collect_diagnostics("build_pipeline", out_dir, "main_exception", pid=os.getpid())
        fail_diag["reason"] = "pipeline_exception"
        fail_diag["exception"] = str(e)
        fail_diag_path = out_dir / "diagnostics_pipeline_exception.json"
        write_diagnostics(fail_diag_path, fail_diag)
        manifest["diagnostics"].append(str(fail_diag_path))

        manifest_path = out_dir / "run_manifest.json"
        dump_json(manifest_path, manifest)
        logger.error("Pipeline failed: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
