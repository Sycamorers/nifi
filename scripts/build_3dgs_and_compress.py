#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.gs import CompressionConfig, HACPPWrapper, Proxy3DGSCompressor



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 3DGS artifacts and compress scenes")
    p.add_argument("--scene", type=str, required=True, help="Path to dataset scene directory")
    p.add_argument("--rates", type=float, nargs="+", required=True, help="Compression rates")
    p.add_argument("--out", type=str, required=True, help="Artifact output directory")
    p.add_argument("--method", type=str, default="proxy", choices=["proxy", "hacpp"])

    p.add_argument("--holdout_every", type=int, default=8)
    p.add_argument("--max_images", type=int, default=None)

    p.add_argument("--jpeg_quality_min", type=int, default=8)
    p.add_argument("--jpeg_quality_max", type=int, default=55)
    p.add_argument("--downsample_min", type=int, default=2)
    p.add_argument("--downsample_max", type=int, default=8)

    p.add_argument("--hacpp_repo", type=str, default=None)
    p.add_argument("--hacpp_train_cmd", type=str, default="")
    p.add_argument("--hacpp_compress_cmd", type=str, default="")
    return p.parse_args()



def run_proxy(args: argparse.Namespace) -> None:
    cfg = CompressionConfig(
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        downsample_min=args.downsample_min,
        downsample_max=args.downsample_max,
    )
    compressor = Proxy3DGSCompressor(cfg)
    meta = compressor.build_scene_artifacts(
        scene_dir=Path(args.scene),
        rates=args.rates,
        out_dir=Path(args.out),
        holdout_every=args.holdout_every,
        max_images=args.max_images,
    )
    print(json.dumps(meta, indent=2))



def run_hacpp(args: argparse.Namespace) -> None:
    if not args.hacpp_repo:
        raise ValueError("--hacpp_repo is required when --method hacpp")
    if not args.hacpp_train_cmd or not args.hacpp_compress_cmd:
        raise ValueError("--hacpp_train_cmd and --hacpp_compress_cmd are required for HAC++")

    wrapper = HACPPWrapper(Path(args.hacpp_repo))
    wrapper.run(
        scene_dir=Path(args.scene),
        out_dir=Path(args.out),
        rates=args.rates,
        train_cmd_template=args.hacpp_train_cmd,
        compress_cmd_template=args.hacpp_compress_cmd,
    )

    meta = {
        "scene_dir": args.scene,
        "rates": args.rates,
        "compression": {"backend": "hacpp", "repo": args.hacpp_repo},
    }
    out_meta = Path(args.out) / "metadata.json"
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] HAC++ metadata written to {out_meta}")



def main() -> None:
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.method == "proxy":
        run_proxy(args)
    else:
        run_hacpp(args)


if __name__ == "__main__":
    main()
