# EXP002: Unified few-shot episode evaluation

## Step 1: Build a shared episode list

```bash
python scripts/build_unified_episodes.py \
  --labels-csv data/splits/silkbehav4_valid1153.csv \
  --split train \
  --episodes 1000 --way 4 --shot 5 --query 5 --seed 42 \
  --output experiments/exp002_fsar_unified/episodes_4w5s5q_seed42.jsonl
```

## Step 2: Run each expert on the same episodes

Each expert must output JSONL rows with fields:
- `episode_id` (int)
- `query_index` (int, 0-based index within that episode's query list)
- `pred` (class name)

## Step 3: Aggregate metrics and decision gates

```bash
python scripts/eval_unified_predictions.py \
  --episodes experiments/exp002_fsar_unified/episodes_4w5s5q_seed42.jsonl \
  --experts trokens=path/to/trokens_preds.jsonl gatev2=path/to/gatev2_preds.jsonl team=path/to/team_preds.jsonl silkmanta=path/to/silkmanta_preds.jsonl \
  --outdir experiments/exp002_fsar_unified/results
```

Output file: `unified_eval_summary.json` with per-expert metrics, disagreement matrix, and oracle gap.
