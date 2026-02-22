"""Utilities for building labeled triptych comparison images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


RGB = Tuple[int, int, int]
Column = Tuple[str, Image.Image]


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _normalize_columns(columns: Sequence[Column]) -> List[Column]:
    if not columns:
        raise ValueError("columns must not be empty")

    base_size = columns[0][1].size
    out: List[Column] = []
    for label, img in columns:
        rgb = img.convert("RGB")
        if rgb.size != base_size:
            raise ValueError(
                f"Image size mismatch for column '{label}': expected {base_size}, found {rgb.size}. "
                "All triptych columns must share the same resolution."
            )
        out.append((label, rgb))
    return out


def build_labeled_triptych(
    columns: Sequence[Column],
    title: str,
    background: RGB = (242, 242, 242),
) -> Image.Image:
    """Build a labeled, fixed-layout triptych image."""
    normalized = _normalize_columns(columns)
    base_w, base_h = normalized[0][1].size
    num_cols = len(normalized)

    pad = 16
    title_h = 34
    label_h = 24

    canvas_w = pad + num_cols * base_w + (num_cols - 1) * pad + pad
    canvas_h = pad + title_h + pad + label_h + pad + base_h + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=background)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(size=17)
    label_font = _load_font(size=14)

    draw.text((pad, pad), title, fill=(20, 20, 20), font=title_font)

    y_label = pad + title_h + pad
    y_img = y_label + label_h + pad
    for idx, (label, img) in enumerate(normalized):
        x = pad + idx * (base_w + pad)
        text_bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((x + (base_w - text_w) // 2, y_label), label, fill=(32, 32, 32), font=label_font)
        canvas.paste(img, (x, y_img))

    return canvas


def save_labeled_triptych(
    *,
    hq_path: Path,
    lq_path: Path,
    restored_path: Path,
    out_path: Path,
    title: str,
    labels: Sequence[str] = ("Original / HQ", "Compressed / LQ", "Restored"),
) -> None:
    """Save a 3-column labeled triptych from input paths."""
    if len(labels) != 3:
        raise ValueError("labels must contain exactly 3 items")

    with Image.open(hq_path) as hq_img, Image.open(lq_path) as lq_img, Image.open(restored_path) as rst_img:
        columns = [
            (labels[0], hq_img.convert("RGB")),
            (labels[1], lq_img.convert("RGB")),
            (labels[2], rst_img.convert("RGB")),
        ]
        canvas = build_labeled_triptych(columns=columns, title=title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
