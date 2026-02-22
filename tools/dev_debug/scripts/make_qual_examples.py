#!/usr/bin/env python3
"""Generate deterministic qualitative comparison mosaics (HQ vs Compressed vs optional Restored)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create qualitative HQ/Compressed(/Restored) mosaics")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g. garden) or scene/artifact path")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: outputs/qual_examples/<scene>")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["hq", "lq", "both"], default="both")
    parser.add_argument("--restore", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rate", type=str, default=None, help="Rate folder or numeric value, e.g. 0.1 or rate_0.100")

    # Optional restore controls (only used when --restore=1)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint for restoration (best.pt/latest.pt)")
    parser.add_argument("--config", type=str, default=None, help="Optional config path for eval_nifi.py")
    return parser.parse_args()



def _is_artifact_scene_dir(path: Path) -> bool:
    return path.is_dir() and (path / "clean").exists() and (path / "metadata.json").exists()



def _resolve_scene_artifacts(scene_arg: str) -> Path:
    candidate = Path(scene_arg)
    if candidate.exists() and _is_artifact_scene_dir(candidate):
        return candidate.resolve()

    if candidate.exists() and candidate.is_dir() and not _is_artifact_scene_dir(candidate):
        out = ROOT / "artifacts" / candidate.name
        if _is_artifact_scene_dir(out):
            return out.resolve()
        raise FileNotFoundError(
            f"Found scene path '{candidate}', but artifact renders are missing at '{out}'. "
            f"Run scripts/build_3dgs_and_compress.py first."
        )

    by_name = ROOT / "artifacts" / scene_arg
    if _is_artifact_scene_dir(by_name):
        return by_name.resolve()

    raise FileNotFoundError(
        f"Could not resolve artifact scene from '{scene_arg}'. "
        f"Expected an artifact directory like artifacts/<scene> with clean/ and rate_*/ images."
    )



def _available_rate_dirs(artifact_scene: Path) -> List[str]:
    return sorted([p.name for p in artifact_scene.glob("rate_*") if p.is_dir()])



def _resolve_rate_dir(artifact_scene: Path, requested_rate: Optional[str]) -> str:
    available = _available_rate_dirs(artifact_scene)
    if not available:
        raise RuntimeError(f"No rate_* directories found under {artifact_scene}")

    if requested_rate is None:
        return available[0]

    s = str(requested_rate).strip()
    if s in available:
        return s

    if not s.startswith("rate_"):
        try:
            s = f"rate_{float(s):.3f}"
        except ValueError:
            pass

    if s in available:
        return s

    raise ValueError(f"Requested rate '{requested_rate}' not found. Available: {available}")



def _load_view_names(artifact_scene: Path, rate_dir: str) -> Tuple[List[str], bool]:
    clean_dir = artifact_scene / "clean"
    lq_dir = artifact_scene / rate_dir

    clean_names = {p.name for p in clean_dir.glob("*.png")}
    lq_names = {p.name for p in lq_dir.glob("*.png")}
    shared = sorted(clean_names & lq_names)
    if not shared:
        raise RuntimeError(f"No shared PNG views between {clean_dir} and {lq_dir}")

    meta_path = artifact_scene / "metadata.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text())
            splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
            test_views = splits.get("test", []) if isinstance(splits, dict) else []
            if isinstance(test_views, list):
                test_filtered = [name for name in test_views if name in clean_names and name in lq_names]
                if test_filtered:
                    return test_filtered, True
        except Exception:
            pass

    return shared, False



def _select_evenly(names: Sequence[str], n: int, seed: int, rotate: bool) -> List[str]:
    if n <= 0:
        raise ValueError("num_views must be > 0")
    items = list(names)
    if not items:
        return []

    if rotate:
        offset = int(seed) % len(items)
        items = items[offset:] + items[:offset]

    if n >= len(items):
        return items

    if n == 1:
        return [items[0]]

    out: List[str] = []
    used = set()
    span = len(items) - 1
    for i in range(n):
        idx = round(i * span / (n - 1))
        name = items[idx]
        if name not in used:
            out.append(name)
            used.add(name)

    if len(out) < n:
        for name in items:
            if name not in used:
                out.append(name)
                used.add(name)
            if len(out) == n:
                break

    return out



def _parse_view_index(name: str, fallback: int) -> int:
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return fallback



def _find_default_ckpt() -> Optional[Path]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return None

    candidates = list(runs_dir.glob("**/best.pt")) + list(runs_dir.glob("**/latest.pt"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]



def _run_restoration(
    artifact_scene: Path,
    rate_dir: str,
    selected_names: List[str],
    out_dir: Path,
    ckpt: Optional[str],
    config: Optional[str],
) -> Dict[str, Path]:
    ckpt_path = Path(ckpt) if ckpt else _find_default_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise RuntimeError(
            "--restore=1 requested but no checkpoint found. "
            "Pass --ckpt /path/to/best.pt (or place one under runs/*/best.pt)."
        )

    pairs_root = out_dir / "_tmp_pairs"
    scene_name = artifact_scene.name
    tmp_scene = pairs_root / scene_name / rate_dir / "test"
    clean_dst = tmp_scene / "clean"
    lq_dst = tmp_scene / "degraded"
    clean_dst.mkdir(parents=True, exist_ok=True)
    lq_dst.mkdir(parents=True, exist_ok=True)

    for name in selected_names:
        shutil.copy2(artifact_scene / "clean" / name, clean_dst / name)
        shutil.copy2(artifact_scene / rate_dir / name, lq_dst / name)

    restore_root = out_dir / "_tmp_restore"
    metrics_path = restore_root / "metrics.json"
    restored_dir = restore_root / "restored"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_nifi.py"),
        "--ckpt",
        str(ckpt_path),
        "--data_root",
        str(pairs_root),
        "--split",
        "test",
        "--out",
        str(metrics_path),
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--save_restored",
        "--restored_dir",
        str(restored_dir),
    ]
    if config:
        cfg_path = Path(config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        cmd.extend(["--config", str(cfg_path)])

    print(f"[info] Running restoration: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Restoration failed. Ensure dependencies/checkpoint are available, or rerun with --restore 0."
        ) from exc

    restored_paths: Dict[str, Path] = {}
    src_root = restored_dir / scene_name / rate_dir
    for name in selected_names:
        src = src_root / name
        if not src.exists():
            raise RuntimeError(f"Restored output missing: {src}")
        restored_paths[name] = src

    return restored_paths



def _save_standardized(src: Path, dst: Path, target_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        if target_size is not None and rgb.size != target_size:
            rgb = rgb.resize(target_size, resample=Image.Resampling.BICUBIC)
        rgb.save(dst)
        return rgb.size



def _make_mosaic(
    columns: List[Tuple[str, Path]],
    title: str,
    out_path: Path,
) -> None:
    loaded: List[Tuple[str, Image.Image]] = []
    for label, path in columns:
        with Image.open(path) as img:
            loaded.append((label, img.convert("RGB")))

    base_w, base_h = loaded[0][1].size
    normalized = []
    for label, img in loaded:
        if img.size != (base_w, base_h):
            img = img.resize((base_w, base_h), resample=Image.Resampling.BICUBIC)
        normalized.append((label, img))

    pad = 14
    title_h = 28
    label_h = 22
    cols = len(normalized)

    canvas_w = pad + cols * base_w + (cols - 1) * pad + pad
    canvas_h = pad + title_h + pad + label_h + pad + base_h + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(236, 236, 236))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, pad), title, fill=(20, 20, 20), font=font)

    y_label = pad + title_h + pad
    y_img = y_label + label_h + pad
    for i, (label, img) in enumerate(normalized):
        x = pad + i * (base_w + pad)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((x + (base_w - text_w) // 2, y_label), label, fill=(30, 30, 30), font=font)
        canvas.paste(img, (x, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)



def main() -> None:
    args = parse_args()

    artifact_scene = _resolve_scene_artifacts(args.scene)
    scene_name = artifact_scene.name
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "qual_examples" / scene_name
    rate_dir = _resolve_rate_dir(artifact_scene, args.rate)

    candidate_names, used_canonical_test = _load_view_names(artifact_scene, rate_dir)
    selected = _select_evenly(
        candidate_names,
        n=int(args.num_views),
        seed=int(args.seed),
        rotate=not used_canonical_test,
    )

    print(f"[info] scene: {scene_name}")
    print(f"[info] artifact dir: {artifact_scene}")
    print(f"[info] rate dir: {rate_dir}")
    print(f"[info] canonical test split used: {used_canonical_test}")

    indexed = [(_parse_view_index(name, i), name) for i, name in enumerate(selected)]
    print(f"[info] selected view indices/names: {indexed}")

    hq_dir = out_dir / "hq"
    lq_dir = out_dir / "lq"
    restored_dir = out_dir / "restored"
    mosaic_dir = out_dir / "mosaic"

    for directory in [hq_dir, lq_dir, restored_dir, mosaic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    restore_enabled = int(args.restore) == 1
    if restore_enabled and args.mode != "both":
        raise ValueError("--restore=1 is supported only with --mode both")

    restored_paths: Dict[str, Path] = {}
    if restore_enabled:
        restored_paths = _run_restoration(
            artifact_scene=artifact_scene,
            rate_dir=rate_dir,
            selected_names=selected,
            out_dir=out_dir,
            ckpt=args.ckpt,
            config=args.config,
        )

    target_size: Optional[Tuple[int, int]] = None
    if restore_enabled and selected:
        # Keep comparison fair: avoid upsampling restored outputs (often generated at model eval size).
        with Image.open(restored_paths[selected[0]]) as probe:
            target_size = probe.convert("RGB").size

    manifest = {
        "scene": scene_name,
        "artifact_scene": str(artifact_scene),
        "rate_dir": rate_dir,
        "mode": args.mode,
        "restore": restore_enabled,
        "seed": int(args.seed),
        "selected_views": [],
    }

    for k, name in enumerate(selected):
        hq_src = artifact_scene / "clean" / name
        lq_src = artifact_scene / rate_dir / name

        view_name = f"view_{k:03d}.png"
        hq_out = hq_dir / view_name
        lq_out = lq_dir / view_name
        restored_out = restored_dir / view_name

        if args.mode in {"hq", "both"}:
            size = _save_standardized(hq_src, hq_out, target_size)
            if target_size is None:
                target_size = size

        if args.mode in {"lq", "both"}:
            if target_size is None:
                size = _save_standardized(lq_src, lq_out, target_size)
                target_size = size
            else:
                _save_standardized(lq_src, lq_out, target_size)

        if restore_enabled:
            _save_standardized(restored_paths[name], restored_out, target_size)

        columns: List[Tuple[str, Path]] = []
        if args.mode in {"hq", "both"}:
            columns.append(("HQ", hq_out))
        if args.mode in {"lq", "both"}:
            columns.append(("Compressed", lq_out))
        if restore_enabled:
            columns.append(("Restored", restored_out))

        title = f"Scene: {scene_name} | Rate: {rate_dir} | Source view: {name}"
        _make_mosaic(columns, title=title, out_path=mosaic_dir / view_name)

        manifest["selected_views"].append(
            {
                "k": k,
                "source_name": name,
                "source_index": _parse_view_index(name, k),
                "mosaic": str((mosaic_dir / view_name).resolve()),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote qualitative examples to: {out_dir}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
