#!/usr/bin/env python3
import argparse
import json
import math
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.utils.diagnostics import collect_diagnostics, dump_stack_traces, write_diagnostics
from nifi.utils.logging import dump_json, get_logger


@dataclass
class RetryProfile:
    name: str
    description: str
    env_updates: Dict[str, str] = field(default_factory=dict)
    build_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    success: bool
    return_code: int
    reason: str
    elapsed_sec: float
    log_path: Path
    diagnostics_path: Optional[Path] = None
    oom_detected: bool = False



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-test harness with hang diagnostics and auto-fallback")
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[0.5])
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cpu", action="store_true")

    p.add_argument("--heartbeat_sec", type=int, default=10)
    p.add_argument("--idle_timeout_sec", type=int, default=60)
    p.add_argument("--timeout_stage_sec", type=int, default=1800)
    p.add_argument("--max_retries", type=int, default=2, help="Retries per step (in addition to initial attempt)")
    return p.parse_args()



def stream_process_output(proc: subprocess.Popen, q: queue.Queue, stream_name: str) -> None:
    pipe = proc.stdout if stream_name == "stdout" else proc.stderr
    assert pipe is not None
    for line in iter(pipe.readline, ""):
        if line == "":
            break
        q.put((stream_name, line))
    q.put((stream_name, None))



def run_step_once(
    name: str,
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    out_dir: Path,
    idle_timeout_sec: int,
    timeout_sec: int,
    heartbeat_sec: int,
    logger,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with log_path.open("w") as logf:
        logf.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        q: queue.Queue = queue.Queue()
        t_out = threading.Thread(target=stream_process_output, args=(proc, q, "stdout"), daemon=True)
        t_err = threading.Thread(target=stream_process_output, args=(proc, q, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        done_streams = 0
        last_output_ts = time.time()
        last_heartbeat_ts = time.time()
        oom_detected = False

        while True:
            now = time.time()
            elapsed = now - start

            try:
                stream_name, payload = q.get(timeout=1.0)
                if payload is None:
                    done_streams += 1
                else:
                    line = payload.rstrip("\n")
                    last_output_ts = now
                    if "out of memory" in line.lower() or "cuda oom" in line.lower():
                        oom_detected = True
                    prefix = f"[{name}:{stream_name}] "
                    print(prefix + line)
                    logf.write(prefix + line + "\n")
                    logf.flush()
            except queue.Empty:
                pass

            if now - last_heartbeat_ts >= float(heartbeat_sec):
                logger.info(
                    "[selftest:heartbeat] step=%s elapsed=%.1fs idle=%.1fs",
                    name,
                    elapsed,
                    now - last_output_ts,
                )
                last_heartbeat_ts = now

            if proc.poll() is not None and done_streams >= 2:
                break

            if elapsed > float(timeout_sec):
                reason = f"timeout>{timeout_sec}s"
                logger.error("Step %s exceeded timeout. Collecting diagnostics...", name)
                dump_stack_traces(proc.pid)
                time.sleep(1.0)

                diag = collect_diagnostics(name, out_dir, "selftest_timeout", pid=proc.pid)
                diag["reason"] = reason
                diag["command"] = cmd
                diag_path = log_path.with_suffix(".diag.json")
                write_diagnostics(diag_path, diag)

                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return StepResult(False, -9, reason, time.time() - start, log_path, diag_path, oom_detected)

            idle = now - last_output_ts
            if idle > float(idle_timeout_sec):
                reason = f"idle>{idle_timeout_sec}s"
                logger.error("Step %s produced no output for %.1fs. Collecting diagnostics...", name, idle)
                dump_stack_traces(proc.pid)
                time.sleep(1.0)

                diag = collect_diagnostics(name, out_dir, "selftest_idle_timeout", pid=proc.pid)
                diag["reason"] = reason
                diag["command"] = cmd
                diag_path = log_path.with_suffix(".diag.json")
                write_diagnostics(diag_path, diag)

                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return StepResult(False, -9, reason, time.time() - start, log_path, diag_path, oom_detected)

        rc = int(proc.returncode)
        elapsed = time.time() - start
        if rc == 0:
            return StepResult(True, rc, "ok", elapsed, log_path, None, oom_detected)

        reason = f"exit_code={rc}"
        diag_path = log_path.with_suffix(".diag.json")
        diag = collect_diagnostics(name, out_dir, "step_failed", pid=os.getpid())
        diag["reason"] = reason
        diag["command"] = cmd
        write_diagnostics(diag_path, diag)
        return StepResult(False, rc, reason, elapsed, log_path, diag_path, oom_detected)



def build_profiles(args: argparse.Namespace) -> List[RetryProfile]:
    profiles = [
        RetryProfile(
            name="baseline",
            description="No fallback; run requested settings",
            env_updates={},
            build_overrides={},
        ),
        RetryProfile(
            name="threads_single_process",
            description="Reduce CPU threading + force single-process execution",
            env_updates={
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
            },
            build_overrides={},
        ),
        RetryProfile(
            name="reduced_workload_isolation",
            description=(
                "Reduce workload (images/views/steps/resolution), bypass compression to isolate train->render path, "
                "and use tiny checkpoint path"
            ),
            env_updates={
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            build_overrides={
                "max_images": 6,
                "max_steps": 20,
                "render_max_views": 2,
                "render_short_side": 256,
                "skip_compress": True,
            },
        ),
    ]

    if args.use_cpu:
        for p in profiles:
            p.env_updates["CUDA_VISIBLE_DEVICES"] = ""
    else:
        for p in profiles:
            p.env_updates.setdefault("CUDA_VISIBLE_DEVICES", "0")

    return profiles



def build_build_cmd(args: argparse.Namespace, overrides: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/build_3dgs_and_compress.py",
        "--scene",
        args.scene,
        "--out",
        args.out,
        "--rates",
        *[str(x) for x in args.rates],
        "--seed",
        str(args.seed),
        "--timeout_stage_sec",
        str(args.timeout_stage_sec),
        "--heartbeat_sec",
        str(args.heartbeat_sec),
        "--idle_timeout_sec",
        str(args.idle_timeout_sec),
    ]

    if args.smoke:
        cmd.append("--smoke")
    if args.use_cpu:
        cmd.append("--use_cpu")

    mapping = {
        "max_images": "--max_images",
        "max_steps": "--max_steps",
        "render_max_views": "--render_max_views",
        "render_short_side": "--render_short_side",
    }
    for k, flag in mapping.items():
        if k in overrides and overrides[k] is not None:
            cmd.extend([flag, str(overrides[k])])

    if overrides.get("skip_compress", False):
        cmd.append("--skip_compress")

    return cmd



def build_render_cmd(args: argparse.Namespace, pairs_root: Path) -> List[str]:
    return [
        sys.executable,
        "scripts/render_pairs.py",
        "--scene",
        args.out,
        "--split",
        "all",
        "--out",
        str(pairs_root),
    ]



def build_eval_cmd(args: argparse.Namespace, pairs_root: Path, metrics_out: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "nifi.metrics.eval_pairs",
        "--data_root",
        str(pairs_root),
        "--split",
        "test",
        "--pred_key",
        "degraded",
        "--out",
        str(metrics_out),
        "--seed",
        str(args.seed),
    ]
    if args.use_cpu:
        cmd.extend(["--device", "cpu"])
    return cmd



def run_step_with_retries(
    step_name: str,
    cmd_builder,
    args: argparse.Namespace,
    cwd: Path,
    log_dir: Path,
    out_dir: Path,
    profiles: List[RetryProfile],
    logger,
) -> Tuple[StepResult, RetryProfile]:
    max_attempts = min(len(profiles), int(args.max_retries) + 1)

    last_result: Optional[StepResult] = None
    chosen_profile: Optional[RetryProfile] = None

    for attempt in range(max_attempts):
        profile = profiles[attempt]
        chosen_profile = profile

        step_log = log_dir / f"{step_name}.attempt{attempt}.log"
        env = os.environ.copy()
        env.update(profile.env_updates)

        cmd = cmd_builder(profile)
        logger.info(
            "[selftest] step=%s attempt=%d/%d profile=%s desc=%s",
            step_name,
            attempt + 1,
            max_attempts,
            profile.name,
            profile.description,
        )
        logger.info("[selftest] cmd=%s", " ".join(shlex.quote(x) for x in cmd))

        result = run_step_once(
            name=f"{step_name}[{profile.name}]",
            cmd=cmd,
            env=env,
            cwd=cwd,
            log_path=step_log,
            out_dir=out_dir,
            idle_timeout_sec=int(args.idle_timeout_sec),
            timeout_sec=int(args.timeout_stage_sec),
            heartbeat_sec=int(args.heartbeat_sec),
            logger=logger,
        )
        last_result = result

        if result.success:
            logger.info(
                "[selftest] step=%s profile=%s success elapsed=%.2fs",
                step_name,
                profile.name,
                result.elapsed_sec,
            )
            return result, profile

        # OOM-aware override for next profile (strategy 6).
        if result.oom_detected and attempt + 1 < max_attempts:
            logger.warning("OOM detected in step %s, applying OOM downgrade for next retry", step_name)
            profiles[attempt + 1].build_overrides.update(
                {
                    "max_images": 4,
                    "max_steps": 20,
                    "render_max_views": 1,
                    "render_short_side": 192,
                }
            )

        logger.warning(
            "[selftest] step=%s attempt=%d failed reason=%s log=%s",
            step_name,
            attempt + 1,
            result.reason,
            result.log_path,
        )

    assert last_result is not None and chosen_profile is not None
    return last_result, chosen_profile



def find_valid_png(root: Path) -> Optional[Path]:
    for p in sorted(root.rglob("*.png")):
        try:
            if p.stat().st_size <= 0:
                continue
            with Image.open(p) as img:
                img.verify()
            return p
        except Exception:
            continue
    return None



def validate_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file missing: {metrics_path}")
    with metrics_path.open("r") as f:
        payload = json.load(f)

    agg = payload.get("aggregate", payload.get("metrics", {}).get("aggregate", {}))
    if not isinstance(agg, dict):
        raise RuntimeError("metrics aggregate section is missing or invalid")

    numeric = {}
    for k, v in agg.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            numeric[k] = float(v)

    if not numeric:
        raise RuntimeError(f"No finite numeric aggregate metrics found in {metrics_path}")

    return numeric



def main() -> None:
    args = parse_args()
    logger = get_logger("nifi.selftest")

    cwd = ROOT
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    scene_name = Path(args.scene).name
    pairs_root = out_dir / "pairs"
    metrics_out = out_dir / "metrics.json"

    profiles = build_profiles(args)

    manifest: Dict[str, Any] = {
        "command": " ".join(sys.argv),
        "scene": args.scene,
        "out": str(out_dir),
        "rates": [float(r) for r in args.rates],
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "steps": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def build_cmd_with_profile(profile: RetryProfile) -> List[str]:
        return build_build_cmd(args, profile.build_overrides)

    def render_cmd_with_profile(_profile: RetryProfile) -> List[str]:
        return build_render_cmd(args, pairs_root)

    def eval_cmd_with_profile(_profile: RetryProfile) -> List[str]:
        return build_eval_cmd(args, pairs_root, metrics_out)

    # Step 1: build/compress/render degraded.
    build_result, build_profile = run_step_with_retries(
        step_name="build_3dgs_and_compress",
        cmd_builder=build_cmd_with_profile,
        args=args,
        cwd=cwd,
        log_dir=log_dir,
        out_dir=out_dir,
        profiles=[RetryProfile(**vars(p)) for p in profiles],
        logger=logger,
    )
    manifest["steps"]["build_3dgs_and_compress"] = {
        "success": build_result.success,
        "reason": build_result.reason,
        "elapsed_sec": build_result.elapsed_sec,
        "profile": build_profile.name,
        "log": str(build_result.log_path),
        "diagnostics": str(build_result.diagnostics_path) if build_result.diagnostics_path else None,
    }
    if not build_result.success:
        manifest["status"] = "failed"
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        dump_json(out_dir / "selftest_manifest.json", manifest)
        raise SystemExit(1)

    # Step 2: render pairs.
    render_result, render_profile = run_step_with_retries(
        step_name="render_pairs",
        cmd_builder=render_cmd_with_profile,
        args=args,
        cwd=cwd,
        log_dir=log_dir,
        out_dir=out_dir,
        profiles=[RetryProfile(**vars(p)) for p in profiles],
        logger=logger,
    )
    manifest["steps"]["render_pairs"] = {
        "success": render_result.success,
        "reason": render_result.reason,
        "elapsed_sec": render_result.elapsed_sec,
        "profile": render_profile.name,
        "log": str(render_result.log_path),
        "diagnostics": str(render_result.diagnostics_path) if render_result.diagnostics_path else None,
    }
    if not render_result.success:
        manifest["status"] = "failed"
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        dump_json(out_dir / "selftest_manifest.json", manifest)
        raise SystemExit(1)

    # Step 3: metrics evaluation.
    eval_result, eval_profile = run_step_with_retries(
        step_name="eval_pairs",
        cmd_builder=eval_cmd_with_profile,
        args=args,
        cwd=cwd,
        log_dir=log_dir,
        out_dir=out_dir,
        profiles=[RetryProfile(**vars(p)) for p in profiles],
        logger=logger,
    )
    manifest["steps"]["eval_pairs"] = {
        "success": eval_result.success,
        "reason": eval_result.reason,
        "elapsed_sec": eval_result.elapsed_sec,
        "profile": eval_profile.name,
        "log": str(eval_result.log_path),
        "diagnostics": str(eval_result.diagnostics_path) if eval_result.diagnostics_path else None,
    }
    if not eval_result.success:
        manifest["status"] = "failed"
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        dump_json(out_dir / "selftest_manifest.json", manifest)
        raise SystemExit(1)

    # Final assertions.
    valid_png = find_valid_png(out_dir)
    if valid_png is None:
        manifest["status"] = "failed"
        manifest["error"] = "No readable PNG found under selftest output"
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        dump_json(out_dir / "selftest_manifest.json", manifest)
        raise SystemExit(1)

    numeric_metrics = validate_metrics(metrics_out)

    manifest["status"] = "pass"
    manifest["scene_name"] = scene_name
    manifest["valid_png"] = str(valid_png)
    manifest["metrics_json"] = str(metrics_out)
    manifest["metrics_csv"] = str(metrics_out.with_suffix(".csv"))
    manifest["numeric_metrics"] = numeric_metrics
    manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump_json(out_dir / "selftest_manifest.json", manifest)

    logger.info("PASS | png=%s | metrics=%s | numeric=%s", valid_png, metrics_out, numeric_metrics)
    print("PASS")
    print(f"  valid_png: {valid_png}")
    print(f"  metrics_json: {metrics_out}")
    print(f"  metrics_csv: {metrics_out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
