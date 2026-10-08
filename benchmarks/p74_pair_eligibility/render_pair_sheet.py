#!/usr/bin/env python3
"""Render a compact visual audit sheet for a P74 pair manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CELL_WIDTH = 960
IMAGE_HEIGHT = 290
HEADER_HEIGHT = 58
CELL_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT
GAP = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="P74 pair-selection visual audit")
    parser.add_argument("--columns", type=int, default=3)
    return parser.parse_args()


def short_view(view_id: str) -> str:
    return view_id.rsplit("+", 1)[-1]


def fields(row: dict) -> tuple[str, float, float | None]:
    if "scene_cloud_intersection" in row:
        status = row["selection"]["status"]
        cloud = row["difficulty_reference"]["value"]
        covisible = row["symmetric_covisible_fraction"]
    else:
        status = row["eligibility_status"]
        cloud = row["scene_cloud_overlap"]
        covisible = row.get("symmetric_covisible_fraction")
    return status, float(cloud), None if covisible is None else float(covisible)


def resize_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ImageOps.fit(
            image,
            size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def render_cell(row: dict) -> Image.Image:
    cell = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(cell)
    font = ImageFont.load_default()
    status, cloud, covisible = fields(row)
    left_name = short_view(row["from_view_id"])
    right_name = short_view(row["to_view_id"])
    metric = f"cloud overlap={cloud:.3f}"
    if covisible is not None:
        metric += f" | co-visible={covisible:.3f}"
    draw.text((12, 8), f"{left_name} -> {right_name}", fill="#111827", font=font)
    draw.text((12, 29), f"{status} | {metric}", fill="#374151", font=font)
    half = (CELL_WIDTH - GAP) // 2
    left = resize_panel(Path(row["from_rgb_path"]), (half, IMAGE_HEIGHT))
    right = resize_panel(Path(row["to_rgb_path"]), (half, IMAGE_HEIGHT))
    cell.paste(left, (0, HEADER_HEIGHT))
    cell.paste(right, (half + GAP, HEADER_HEIGHT))
    draw.rectangle((half, HEADER_HEIGHT, half + GAP, CELL_HEIGHT), fill="#111827")
    draw.text((8, CELL_HEIGHT - 18), "FROM", fill="white", font=font)
    draw.text((half + GAP + 8, CELL_HEIGHT - 18), "TO", fill="white", font=font)
    return cell


def main() -> None:
    args = parse_args()
    if args.columns < 1:
        raise ValueError("columns must be >= 1")
    rows = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("records file is empty")
    row_count = math.ceil(len(rows) / args.columns)
    title_height = 48
    sheet = Image.new(
        "RGB",
        (args.columns * CELL_WIDTH, title_height + row_count * CELL_HEIGHT),
        "#e5e7eb",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 16), args.title, fill="#111827", font=ImageFont.load_default())
    for index, row in enumerate(rows):
        x = index % args.columns * CELL_WIDTH
        y = title_height + index // args.columns * CELL_HEIGHT
        sheet.paste(render_cell(row), (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
