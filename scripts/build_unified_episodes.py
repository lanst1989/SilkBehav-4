#!/usr/bin/env python3
"""Build unified few-shot episode list for SilkBehav-4.

Output format (JSONL):
{
  "episode_id": 0,
  "classes": ["feeding", ...],
  "support": [{"class": "feeding", "video": "..."}, ...],
  "query": [{"class": "feeding", "video": "..."}, ...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_pool(csv_path: Path, split: str, class_names: list[str]) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            label = row.get("label") or row.get("label_main")
            video = row.get("video") or row.get("filepath") or row.get("clip_path")
            if label in class_names and video:
                pool[label].append(video)

    missing = [c for c in class_names if len(pool[c]) == 0]
    if missing:
        raise ValueError(f"No videos found for classes in split '{split}': {missing}")
    return pool


def sample_episode(
    rng: random.Random,
    pool: dict[str, list[str]],
    class_names: list[str],
    way: int,
    shot: int,
    query: int,
) -> dict:
    selected_classes = rng.sample(class_names, way)
    support = []
    query_items = []
    for cls in selected_classes:
        candidates = pool[cls]
        needed = shot + query
        if len(candidates) < needed:
            raise ValueError(
                f"Class '{cls}' has {len(candidates)} videos, but needs at least {needed}"
            )
        picked = rng.sample(candidates, needed)
        support.extend({"class": cls, "video": v} for v in picked[:shot])
        query_items.extend({"class": cls, "video": v} for v in picked[shot:])

    rng.shuffle(support)
    rng.shuffle(query_items)
    return {"classes": selected_classes, "support": support, "query": query_items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--classes", nargs="+", default=["feeding", "head_swing", "inactive", "locomotion"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--way", type=int, default=4)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = load_pool(args.labels_csv, args.split, args.classes)
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for eid in range(args.episodes):
            ep = sample_episode(rng, pool, args.classes, args.way, args.shot, args.query)
            ep["episode_id"] = eid
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"Wrote {args.episodes} episodes to {args.output}")


if __name__ == "__main__":
    main()
