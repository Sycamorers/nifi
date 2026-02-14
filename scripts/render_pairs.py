#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render aligned clean/degraded pairs from compression artifacts")
    p.add_argument("--scene", type=str, required=True, help="Artifact scene directory")
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    p.add_argument("--out", type=str, required=True, help="Output root directory for pair data")
    return p.parse_args()



def resolve_scene_output(out: Path, scene_name: str) -> Path:
    out = Path(out)
    if out.name in {"train", "test"}:
        base = out.parent
    else:
        base = out

    if base.name == scene_name:
        return base
    return base / scene_name



def ensure_split_names(meta: Dict[str, object], clean_dir: Path) -> Dict[str, List[str]]:
    if "splits" in meta and isinstance(meta["splits"], dict):
        splits = meta["splits"]
        if "train" in splits and "test" in splits:
            return {
                "train": list(splits["train"]),
                "test": list(splits["test"]),
            }

    # fallback split if metadata has no train/test lists
    names = sorted([p.name for p in clean_dir.glob("*.png")])
    train = [n for i, n in enumerate(names) if i % 8 != 0]
    test = [n for i, n in enumerate(names) if i % 8 == 0]
    return {"train": train, "test": test}



def copy_pairs(artifact_scene_dir: Path, scene_out: Path, split: str) -> Dict[str, int]:
    meta_path = artifact_scene_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json missing: {meta_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    clean_dir = artifact_scene_dir / "clean"
    if not clean_dir.exists():
        raise FileNotFoundError(f"clean directory missing: {clean_dir}")

    rates = meta.get("rates", [])
    if not rates:
        # fallback scan rate_* dirs
        rates = [float(p.name.split("rate_")[-1]) for p in artifact_scene_dir.glob("rate_*") if p.is_dir()]

    splits = ensure_split_names(meta, clean_dir)
    split_names = [split] if split in {"train", "test"} else ["train", "test"]

    stats = {}
    for rate in rates:
        rate_name = f"rate_{float(rate):.3f}"
        degraded_dir = artifact_scene_dir / rate_name
        if not degraded_dir.exists():
            raise FileNotFoundError(f"degraded rate directory missing: {degraded_dir}")

        for s in split_names:
            clean_out = scene_out / rate_name / s / "clean"
            degraded_out = scene_out / rate_name / s / "degraded"
            clean_out.mkdir(parents=True, exist_ok=True)
            degraded_out.mkdir(parents=True, exist_ok=True)

            copied = 0
            for name in splits[s]:
                src_clean = clean_dir / name
                src_degraded = degraded_dir / name
                if not src_clean.exists() or not src_degraded.exists():
                    continue
                shutil.copy2(src_clean, clean_out / name)
                shutil.copy2(src_degraded, degraded_out / name)
                copied += 1

            stats[f"{rate_name}/{s}"] = copied

    with (scene_out / "pairs_manifest.json").open("w") as f:
        json.dump({"scene": artifact_scene_dir.name, "stats": stats}, f, indent=2)

    return stats



def main() -> None:
    args = parse_args()
    artifact_scene_dir = Path(args.scene)
    if not artifact_scene_dir.exists():
        raise FileNotFoundError(f"Artifact scene path not found: {artifact_scene_dir}")

    scene_name = artifact_scene_dir.name
    scene_out = resolve_scene_output(Path(args.out), scene_name)
    scene_out.mkdir(parents=True, exist_ok=True)

    stats = copy_pairs(artifact_scene_dir, scene_out, args.split)
    print(json.dumps({"scene_out": str(scene_out), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
