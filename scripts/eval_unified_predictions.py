#!/usr/bin/env python3
"""Aggregate expert predictions on unified episodes and compute key metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def flatten_labels(episodes_path: Path):
    y_true = []
    keys = []
    for ep in read_jsonl(episodes_path):
        eid = ep["episode_id"]
        for i, item in enumerate(ep["query"]):
            keys.append((eid, i))
            y_true.append(item["class"])
    return keys, y_true


def load_preds(path: Path):
    # expected per line: {"episode_id": int, "query_index": int, "pred": "class_name"}
    out = {}
    for row in read_jsonl(path):
        out[(row["episode_id"], row["query_index"])] = row["pred"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--experts", nargs="+", required=True, help="name=path/to/pred.jsonl")
    p.add_argument("--labels", nargs="+", default=["feeding", "head_swing", "inactive", "locomotion"])
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    keys, y_true = flatten_labels(args.episodes)
    expert_preds = {}
    summary = {}

    for item in args.experts:
        name, pred_path = item.split("=", 1)
        pred_map = load_preds(Path(pred_path))
        y_pred = [pred_map[k] for k in keys]
        expert_preds[name] = y_pred
        summary[name] = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=args.labels)),
            "per_class_f1": {
                label: float(f1_score(y_true, y_pred, labels=[label], average="macro"))
                for label in args.labels
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=args.labels).tolist(),
            "classification_report": classification_report(y_true, y_pred, labels=args.labels, output_dict=True),
        }

    arr_true = np.array(y_true)
    matrix = {}
    names = list(expert_preds.keys())
    for a in names:
        matrix[a] = {}
        arr_a = np.array(expert_preds[a])
        for b in names:
            arr_b = np.array(expert_preds[b])
            matrix[a][b] = float(np.mean(arr_a != arr_b))

    oracle_correct = np.zeros(len(y_true), dtype=bool)
    for n in names:
        oracle_correct |= np.array(expert_preds[n]) == arr_true
    oracle_acc = float(np.mean(oracle_correct))

    best_macro = max(summary[n]["macro_f1"] for n in names)
    summary["_oracle"] = {
        "oracle_accuracy": oracle_acc,
        "best_single_macro_f1": best_macro,
        "oracle_gap_vs_best_macro_f1": oracle_acc - best_macro,
        "disagreement_matrix": matrix,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "unified_eval_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
