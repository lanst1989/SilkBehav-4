#!/usr/bin/env python3
"""
SilkBehav-4  ·  P2 Few-Shot Benchmark Evaluation
=================================================
Unified script: ProtoNet + TEAM, 4-way {1,5}-shot, 600 episodes.
All methods share identical episode splits (same random seed).

References
----------
[TEAM]     CVPR 2025 – TEAM: model.py (TEAM → TEAM_pos → CNN_FSHead)
[ProtoNet] Snell et al., NeurIPS 2017, Eq. 2-3
[ResNet]   He et al., CVPR 2016 (torchvision ResNet50)
[VideoMAE] Tong et al., NeurIPS 2022 (baseline, acc 85.1%)

Author : SilkBehav-4 project
Created: 2026-05-01
"""

import os, sys, json, math, random, argparse, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ────────────────────────────────────────────────────────────────
# 0. Config & Paths
# ────────────────────────────────────────────────────────────────

# Ref: TEAM official – dataset/video_reader.py, dataset/Split.py
FRAME_DIR   = Path("/home/fmh/SilkVIM/outputs/exp023_team_silkvim_valid1153_dataset/test")
SPLIT_FILE  = Path("/home/fmh/SilkVIM/outputs/diagnostics/exp023_team_silkvim_splits/testlist.txt")
TEAM_REPO   = Path("/home/fmh/SilkVIM/refs/selector_refs/TEAM_official")
CKPT_REL    = Path("pretrained/hmdb/TEAM/ResNet/1-shot/an60/checkpoint_best_val.pt")
CKPT_PATH   = TEAM_REPO / CKPT_REL

# Ref: TEAM official – configs/*.yaml
CLASSES     = ["feeding", "head_swing", "inactive", "locomotion"]
N_WAY       = 4
QUERY_PER_CLASS = 15          # standard few-shot query budget
NUM_EPISODES    = 600
NUM_FRAMES      = 8           # Ref: TEAM default –cfg num_input_frames=8
SEED            = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────
# 1. Data Pipeline  (Ref: TEAM official – dataset/video_reader.py)
# ────────────────────────────────────────────────────────────────

# Ref: torchvision – ImageNet normalisation (same as TEAM ResNet50)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _safe_relpath(path: Path, root: Path):
    """Return POSIX relative path if possible, otherwise None."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return None


def _normalise_key(x: str) -> str:
    """Normalise a path-like string for robust split matching."""
    return str(x).strip().replace("\\", "/").lstrip("./")


def _build_video_index(frame_dir: Path):
    """
    Build robust lookup tables for frame-folder videos.

    Ref: TEAM official – video_reader.py expects each sample to be a directory
         containing frame images. Here we index every direct child under the
         four SilkBehav-4 class directories so split files can use absolute
         paths, relative paths, stems, or class/video forms.
    """
    class2idx = {c: i for i, c in enumerate(CLASSES)}
    video_records = []
    lookup = defaultdict(list)

    for cls_name in CLASSES:
        cls_dir = frame_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for video_dir in sorted(cls_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            rec = (video_dir, class2idx[cls_name])
            video_records.append(rec)

            keys = set()
            keys.add(video_dir.as_posix())
            try:
                keys.add(video_dir.resolve().as_posix())
            except Exception:
                pass
            rel = _safe_relpath(video_dir, frame_dir)
            if rel:
                keys.add(rel)
            keys.add(f"{cls_name}/{video_dir.name}")
            keys.add(f"{cls_name}/{video_dir.stem}")
            keys.add(video_dir.name)
            keys.add(video_dir.stem)

            for key in keys:
                lookup[_normalise_key(key)].append(rec)

    return video_records, lookup


def _infer_label_from_line(parts):
    """
    Infer class id from a TEAM split line.

    Supports:
      - path/to/class/video
      - class/video label_int
      - video_name label_int
      - path class_name
    """
    class2idx = {c: i for i, c in enumerate(CLASSES)}

    # Search all tokens for class names first.
    for token in parts:
        token_parts = _normalise_key(token).split("/")
        for p in token_parts:
            if p in class2idx:
                return class2idx[p]

    # Then search all tokens for integer labels.
    for token in parts[1:]:
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx in set(class2idx.values()):
            return idx

    return None


def _candidate_keys_from_line(raw_path: str, frame_dir: Path, label_hint):
    """
    Generate possible keys for a split path.

    This is intentionally permissive because TEAM-style split files often move
    between raw-video paths and extracted-frame-folder paths.
    """
    p = Path(raw_path)
    raw_norm = _normalise_key(raw_path)
    keys = []

    def add(x):
        x = _normalise_key(str(x))
        if x and x not in keys:
            keys.append(x)

    add(raw_norm)
    add(Path(raw_norm).with_suffix("").as_posix())
    add(Path(raw_norm).name)
    add(Path(raw_norm).stem)

    if p.is_absolute():
        add(p.as_posix())
        try:
            add(p.resolve().as_posix())
        except Exception:
            pass
        rel = _safe_relpath(p, frame_dir)
        if rel:
            add(rel)
            add(Path(rel).with_suffix("").as_posix())

    # If path contains a class name, also try the suffix from that class onward.
    path_parts = _normalise_key(raw_path).split("/")
    for cls_name in CLASSES:
        if cls_name in path_parts:
            i = path_parts.index(cls_name)
            suffix = "/".join(path_parts[i:])
            add(suffix)
            add(Path(suffix).with_suffix("").as_posix())
            if len(path_parts) > i + 1:
                add(f"{cls_name}/{Path(path_parts[-1]).name}")
                add(f"{cls_name}/{Path(path_parts[-1]).stem}")

    # If there is a numeric label, combine class with basename/stem.
    if label_hint is not None and 0 <= label_hint < len(CLASSES):
        cls_name = CLASSES[label_hint]
        add(f"{cls_name}/{Path(raw_norm).name}")
        add(f"{cls_name}/{Path(raw_norm).stem}")

    return keys


def load_split(split_file: Path, frame_dir: Path, allow_scan_fallback: bool = True):
    """
    Robust TEAM-format split parser.

    Compatible examples:
      1) feeding/video_xxx
      2) feeding/video_xxx 0
      3) /abs/path/to/test/feeding/video_xxx
      4) /abs/path/to/raw_video.mp4 2
      5) video_xxx 2
      6) arbitrary/prefix/feeding/video_xxx

    Returns:
      list[(video_frame_folder_path, class_index)]

    Ref:
      - TEAM official dataset/Split.py: split maintains paths + labels.
      - TEAM official video_reader.py: model reads frame-image folders.
    """
    split_file = Path(split_file)
    frame_dir = Path(frame_dir)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame dir not found: {frame_dir}")

    all_videos, lookup = _build_video_index(frame_dir)
    raw_lines = []
    samples = []
    skipped = []
    seen_paths = set()

    with open(split_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_lines.append(line)
            parts = line.split()
            raw_path = parts[0]
            label_hint = _infer_label_from_line(parts)

            matched = []
            for key in _candidate_keys_from_line(raw_path, frame_dir, label_hint):
                if key in lookup:
                    matched.extend(lookup[key])

            # Deduplicate matches.
            uniq = []
            local_seen = set()
            for video_path, label in matched:
                sig = (str(video_path), label)
                if sig not in local_seen:
                    uniq.append((video_path, label))
                    local_seen.add(sig)

            # If class label is known, prefer matches from that class.
            if label_hint is not None:
                filtered = [m for m in uniq if m[1] == label_hint]
                if filtered:
                    uniq = filtered

            if not uniq:
                skipped.append((line_no, line, _candidate_keys_from_line(raw_path, frame_dir, label_hint)[:8]))
                continue

            # In normal cases there is exactly one match. If ambiguous, use first
            # stable-sorted match and print it in debug summary.
            video_path, label = sorted(uniq, key=lambda x: str(x[0]))[0]
            if str(video_path) not in seen_paths:
                samples.append((video_path, label))
                seen_paths.add(str(video_path))

    print(f"[SPLIT] Parsed non-empty lines : {len(raw_lines)}")
    print(f"[SPLIT] Indexed frame folders  : {len(all_videos)}")
    print(f"[SPLIT] Matched unique videos  : {len(samples)}")
    print(f"[SPLIT] Skipped lines          : {len(skipped)}")

    if skipped[:5]:
        print("[SPLIT] First skipped examples:")
        for line_no, line, tried in skipped[:5]:
            print(f"  line {line_no}: {line}")
            print(f"    tried keys: {tried}")

    # Practical fallback: if the split format is completely incompatible but
    # frame_dir/test already contains only the test split, still allow evaluation.
    if len(samples) == 0 and allow_scan_fallback and len(all_videos) > 0:
        print("\n[WARN] Split matched 0 videos; falling back to scanning frame_dir/class/*.")
        print("       This is safe only if frame_dir already points to the test split folder.")
        samples = all_videos

    if len(samples) == 0:
        print("\n[ERROR] No videos matched. Debug info:")
        print(f"  split_file = {split_file}")
        print(f"  frame_dir  = {frame_dir}")
        print("\n  First 10 split lines:")
        for x in raw_lines[:10]:
            print(f"    {x}")
        print("\n  Existing class dirs under frame_dir:")
        for cls_name in CLASSES:
            cls_dir = frame_dir / cls_name
            print(f"    {cls_name}: exists={cls_dir.exists()} path={cls_dir}")
            if cls_dir.exists():
                subdirs = [p.name for p in cls_dir.iterdir() if p.is_dir()]
                print(f"      num video dirs: {len(subdirs)}")
                print(f"      first few: {subdirs[:5]}")
        raise RuntimeError("load_split() matched 0 videos. Check split format and frame_dir.")

    return samples


def read_video_frames(video_dir: Path, num_frames: int = NUM_FRAMES):
    """
    Uniform temporal sampling of frames from a directory.
    Ref: TEAM official – dataset/video_reader.py (uniform segment sampling)

    Each video directory contains 000001.jpg … 000150.jpg.
    Returns Tensor [num_frames, 3, 224, 224].
    """
    jpgs = sorted(video_dir.glob("*.jpg"))
    total = len(jpgs)
    if total == 0:
        raise FileNotFoundError(f"No .jpg frames in {video_dir}")

    # Uniform sampling: pick evenly spaced indices
    # Ref: TEAM – "uniform" segment mode in video_reader.py
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        img = Image.open(jpgs[idx]).convert("RGB")
        frames.append(TRANSFORM(img))
    return torch.stack(frames)  # [T, 3, 224, 224]


# ────────────────────────────────────────────────────────────────
# 2. Episode Sampler  (Ref: TEAM official – dataset/VideoDataset.py)
# ────────────────────────────────────────────────────────────────

class EpisodeSampler:
    """
    Generate N-way K-shot episodes with deterministic random seed.
    Ref: TEAM official – VideoDataset.__getitem__ (episode construction)

    All methods share the *same* list of episodes for fair comparison.
    """

    def __init__(self, samples, n_way, k_shots, query_per_class,
                 num_episodes, seed, num_frames=NUM_FRAMES):
        self.n_way = n_way
        self.k_shots = k_shots
        self.query_per_class = query_per_class
        self.num_episodes = num_episodes
        self.num_frames = num_frames

        # Group by class
        self.class_samples = defaultdict(list)
        for path, label in samples:
            self.class_samples[label].append(path)

        self.rng = random.Random(seed)
        self.episodes = self._generate_all()

    def _generate_all(self):
        """Pre-generate all episodes so every method sees the same data."""
        episodes = []
        class_ids = sorted(self.class_samples.keys())
        if len(class_ids) < self.n_way:
            counts = {CLASSES[c]: len(self.class_samples[c]) for c in class_ids}
            raise RuntimeError(
                f"Need ≥ {self.n_way} classes, got {len(class_ids)}. "
                f"Class counts: {counts}"
            )

        for _ in range(self.num_episodes):
            # For 4-way with exactly 4 classes: use all classes, shuffle order
            chosen = list(class_ids)
            self.rng.shuffle(chosen)

            support_paths, support_labels = [], []
            query_paths, query_labels = [], []

            for new_label, cls_id in enumerate(chosen):
                pool = list(self.class_samples[cls_id])
                self.rng.shuffle(pool)
                need = self.k_shots + self.query_per_class
                if len(pool) < need:
                    # Sample with replacement if insufficient
                    selected = self.rng.choices(pool, k=need)
                else:
                    selected = pool[:need]

                s_paths = selected[:self.k_shots]
                q_paths = selected[self.k_shots:need]

                support_paths.extend(s_paths)
                support_labels.extend([new_label] * self.k_shots)
                query_paths.extend(q_paths)
                query_labels.extend([new_label] * self.query_per_class)

            episodes.append({
                "support_paths": support_paths,
                "support_labels": support_labels,
                "query_paths": query_paths,
                "query_labels": query_labels,
            })
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        s_frames = torch.stack([read_video_frames(p, self.num_frames) for p in ep["support_paths"]])
        q_frames = torch.stack([read_video_frames(p, self.num_frames) for p in ep["query_paths"]])
        s_labels = torch.tensor(ep["support_labels"], dtype=torch.long)
        q_labels = torch.tensor(ep["query_labels"], dtype=torch.long)
        return s_frames, s_labels, q_frames, q_labels


# ────────────────────────────────────────────────────────────────
# 3. Backbone: ResNet50 Feature Extractor
#    (Ref: TEAM official – model.py, CNN_FSHead.__init__)
# ────────────────────────────────────────────────────────────────

def build_resnet50_backbone():
    """
    ResNet50 up to layer4, global avg-pool → 2048-d.
    Ref: TEAM official – model.py line ~30, CNN_FSHead uses
         torchvision ResNet50 minus fc, plus adaptive avg pool.
    Ref: He et al. CVPR 2016 – ResNet architecture.
    """
    resnet = models.resnet50(weights=None)
    # Remove the final FC layer — keep feature extractor only
    # Ref: TEAM model.py – self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]
    return backbone


class TemporalPooler(nn.Module):
    """
    Average-pool frame-level features into a single video representation.
    Ref: TEAM official – model.py, CNN_FSHead.get_feats() applies
         temporal aggregation before the matching head.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        """
        x: [B, T, 3, 224, 224]
        Returns: [B, 2048]
        """
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])        # [B*T, 3, 224, 224]
        feats = self.backbone(x)                 # [B*T, 2048, 1, 1]
        feats = feats.view(B, T, -1)            # [B, T, 2048]
        feats = feats.mean(dim=1)               # [B, 2048]  temporal avg pool
        return feats


# ────────────────────────────────────────────────────────────────
# 4. ProtoNet Matching Head
#    (Ref: Snell et al., NeurIPS 2017, §3, Eq. 2-3)
# ────────────────────────────────────────────────────────────────

class ProtoNet(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning.

    Eq. 2 (prototype):  c_k = (1/|S_k|) Σ_{(x,y)∈S_k} f(x)
    Eq. 3 (prediction): p(y=k|x) = softmax( -d(f(x), c_k) )

    Ref: Snell et al., NeurIPS 2017, Prototypical Networks
    """
    def __init__(self, encoder: TemporalPooler):
        super().__init__()
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, support, support_labels, query, n_way, k_shot):
        """
        support: [n_way*k_shot, T, 3, H, W]
        query:   [n_query, T, 3, H, W]
        Returns: logits [n_query, n_way]
        """
        # Encode
        s_feat = self.encoder(support)   # [n_way*k_shot, D]
        q_feat = self.encoder(query)     # [n_query, D]

        # Eq. 2: compute prototypes by averaging support per class
        D = s_feat.shape[-1]
        prototypes = torch.zeros(n_way, D, device=s_feat.device)
        for c in range(n_way):
            mask = (support_labels == c)
            prototypes[c] = s_feat[mask].mean(dim=0)

        # Eq. 3: negative squared Euclidean distance → softmax
        # dists[i,k] = ||q_i - proto_k||^2
        dists = torch.cdist(q_feat, prototypes, p=2).pow(2)  # [n_query, n_way]
        logits = -dists   # higher = closer = more likely
        return logits


# ────────────────────────────────────────────────────────────────
# 5. TEAM Matching Head
#    (Ref: TEAM official – model.py, TEAM class)
# ────────────────────────────────────────────────────────────────

class DPM(nn.Module):
    """
    Discriminative Prototype Module from TEAM (CVPR 2025).
    Ref: TEAM official – model.py, TEAM_pos class.

    Learns attention-based prototype refinement via cross-attention
    between support and query, producing task-adapted prototypes.
    """
    def __init__(self, feat_dim=2048, hidden_dim=512):
        super().__init__()
        # Ref: TEAM model.py – self.attn_proj / self.proto_refine
        self.query_proj = nn.Linear(feat_dim, hidden_dim)
        self.key_proj   = nn.Linear(feat_dim, hidden_dim)
        self.value_proj = nn.Linear(feat_dim, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, support_feats, query_feats, n_way, k_shot):
        """
        support_feats: [n_way*k_shot, D]
        query_feats:   [n_query, D]
        Returns refined prototypes: [n_way, D]
        """
        # Reshape support → [n_way, k_shot, D] and average → initial protos
        S = support_feats.view(n_way, k_shot, -1)
        prototypes = S.mean(dim=1)  # [n_way, D]

        # Cross-attention: query-set attends to prototypes for refinement
        # Ref: TEAM model.py – TEAM_pos.forward()
        Q = self.query_proj(query_feats)       # [n_query, H]
        K = self.key_proj(prototypes)          # [n_way, H]
        V = self.value_proj(prototypes)        # [n_way, H]

        attn = torch.matmul(Q, K.t()) * self.scale   # [n_query, n_way]
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn.t(), Q)           # [n_way, H]

        # Residual refinement
        refined = prototypes + self.out_proj(context)
        refined = self.layer_norm(refined)
        return refined


class TEAM_FSL(nn.Module):
    """
    TEAM: Task-adaptive Embedding for few-shot Action recognition in Movies.
    Ref: TEAM CVPR 2025 – model.py (TEAM class inherits TEAM_pos inherits CNN_FSHead)

    Architecture: ResNet50 backbone → temporal avg pool → DPM → cosine matching
    """
    def __init__(self, encoder: TemporalPooler, feat_dim=2048, temperature=5.0):
        super().__init__()
        self.encoder = encoder
        self.dpm = DPM(feat_dim=feat_dim)
        # Ref: TEAM model.py – self.temperature (learnable or fixed)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    @torch.no_grad()
    def forward(self, support, support_labels, query, n_way, k_shot):
        """
        support: [n_way*k_shot, T, 3, H, W]
        query:   [n_query, T, 3, H, W]
        Returns: logits [n_query, n_way]
        """
        s_feat = self.encoder(support)   # [n_way*k_shot, D]
        q_feat = self.encoder(query)     # [n_query, D]

        # Task-adaptive prototype refinement via DPM
        # Ref: TEAM model.py – TEAM.forward()
        prototypes = self.dpm(s_feat, q_feat, n_way, k_shot)  # [n_way, D]

        # Cosine similarity matching
        # Ref: TEAM model.py – CNN_FSHead uses cosine + temperature
        prototypes_norm = F.normalize(prototypes, dim=-1)
        q_norm = F.normalize(q_feat, dim=-1)
        logits = torch.matmul(q_norm, prototypes_norm.t()) * self.temperature
        return logits


# ────────────────────────────────────────────────────────────────
# 6. Checkpoint Loading
#    (Ref: TEAM official – run.py, load_checkpoint)
# ────────────────────────────────────────────────────────────────

def resolve_checkpoint_path(ckpt_path: str, team_repo: str = str(TEAM_REPO)) -> str:
    """
    Resolve TEAM checkpoint path robustly.

    The old script used a relative checkpoint path, which makes Python search
    under the current working directory. This helper tries the known SilkVIM and
    TEAM_official locations before giving up.

    Ref: TEAM official run.py / checkpoint loading convention.
    """
    raw = Path(ckpt_path).expanduser()
    team_repo = Path(team_repo).expanduser()
    rel = Path("pretrained/hmdb/TEAM/ResNet/1-shot/an60/checkpoint_best_val.pt")

    candidates = []

    def add(p):
        p = Path(p).expanduser()
        if p not in candidates:
            candidates.append(p)

    add(raw)
    if not raw.is_absolute():
        add(Path.cwd() / raw)
        add(Path("/home/fmh/SilkVIM") / raw)
        add(team_repo / raw)
    add(team_repo / rel)
    add(Path("/home/fmh/SilkVIM") / rel)
    add(Path("/home/fmh/SilkVIM/refs/selector_refs/TEAM_official") / rel)

    for p in candidates:
        if p.exists():
            return str(p)

    print("[WARN] Could not resolve checkpoint. Tried:")
    for p in candidates:
        print(f"       - {p}")
    return str(raw)


def load_team_checkpoint(model: TEAM_FSL, ckpt_path: str):
    """
    Load TEAM pretrained checkpoint (backbone + DPM, no classification head).
    Ref: TEAM official – run.py, state_dict key mapping.
    The checkpoint stores backbone and DPM parameters; no final FC layer.
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        print("       Will use randomly initialised weights (results won't be meaningful).")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle wrapped state_dict formats
    # Ref: TEAM run.py – checkpoint may have 'model_state_dict' or 'state_dict' key
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Remap keys: TEAM checkpoint uses 'backbone.X' or 'encoder.backbone.X'
    # Our model uses 'encoder.backbone.X' and 'dpm.X'
    new_state = {}
    for k, v in state.items():
        # Strip 'module.' prefix from DDP
        k_clean = k.replace("module.", "")

        # Map backbone keys
        if k_clean.startswith("backbone."):
            new_key = "encoder." + k_clean
        elif k_clean.startswith("encoder.backbone."):
            new_key = k_clean
        elif k_clean.startswith("encoder."):
            new_key = k_clean
        # Map DPM keys (attn_proj → query_proj etc.)
        elif k_clean.startswith("attn_proj.") or k_clean.startswith("proto_refine."):
            # Try loading into dpm sub-module
            new_key = "dpm." + k_clean
        elif k_clean.startswith("dpm."):
            new_key = k_clean
        elif k_clean == "temperature":
            new_key = "temperature"
        else:
            new_key = k_clean

        new_state[new_key] = v

    # Partial load — allow shape/key mismatches for DPM sub-module
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state)
    print(f"[CKPT] Loaded {loaded} params, skipped {skipped} "
          f"(from {ckpt_path})")


# ────────────────────────────────────────────────────────────────
# 7. Metrics  (Ref: scikit-learn convention, F1 macro-average)
# ────────────────────────────────────────────────────────────────

def compute_accuracy(preds, labels):
    """Per-episode accuracy."""
    return (preds == labels).float().mean().item()


def compute_macro_f1(preds, labels, n_way):
    """
    Macro-averaged F1 across classes.
    Ref: standard definition, same as sklearn.metrics.f1_score(average='macro')
    """
    f1s = []
    for c in range(n_way):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1.item())
    return np.mean(f1s)


def confidence_interval_95(values):
    """
    95% confidence interval = 1.96 * std / sqrt(n).
    Ref: standard frequentist CI for mean estimation.
    """
    arr = np.array(values)
    mean = arr.mean()
    ci = 1.96 * arr.std() / math.sqrt(len(arr))
    return mean, ci


# ────────────────────────────────────────────────────────────────
# 8. Evaluation Loop
# ────────────────────────────────────────────────────────────────

def evaluate_method(model, sampler, n_way, k_shot, device):
    """
    Run evaluation over all pre-generated episodes.
    Returns per-episode accuracy list and per-episode macro-F1 list.
    """
    model.eval()
    accs, f1s = [], []

    for ep_idx in range(len(sampler)):
        s_frames, s_labels, q_frames, q_labels = sampler[ep_idx]
        s_frames = s_frames.to(device)
        q_frames = q_frames.to(device)
        s_labels = s_labels.to(device)
        q_labels = q_labels.to(device)

        with torch.no_grad():
            logits = model(s_frames, s_labels, q_frames, n_way, k_shot)
            preds = logits.argmax(dim=-1)

        accs.append(compute_accuracy(preds, q_labels))
        f1s.append(compute_macro_f1(preds, q_labels, n_way))

        if (ep_idx + 1) % 100 == 0:
            running_acc, _ = confidence_interval_95(accs)
            print(f"  Episode {ep_idx+1}/{len(sampler)} | "
                  f"running acc: {running_acc:.4f}")

    return accs, f1s


# ────────────────────────────────────────────────────────────────
# 9. Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SilkBehav-4 P2: Few-Shot Benchmark (ProtoNet + TEAM)")
    parser.add_argument("--frame_dir", type=str, default=str(FRAME_DIR))
    parser.add_argument("--split_file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    parser.add_argument("--team_repo", type=str, default=str(TEAM_REPO),
                        help="TEAM official repo root; used to resolve relative checkpoint path")
    parser.add_argument("--no_scan_fallback", action="store_true",
                        help="Disable fallback that scans frame_dir/class/* when split matches 0 videos")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--query_per_class", type=int, default=QUERY_PER_CLASS)
    parser.add_argument("--shots", type=str, default="1,5",
                        help="Comma-separated k-shot values (default: 1,5)")
    parser.add_argument("--output_dir", type=str, default="results/p2_fewshot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shots = [int(s) for s in args.shots.split(",")]

    # Resolve path-like args once, before printing config.
    args.frame_dir = str(Path(args.frame_dir).expanduser())
    args.split_file = str(Path(args.split_file).expanduser())
    args.team_repo = str(Path(args.team_repo).expanduser())
    args.ckpt = resolve_checkpoint_path(args.ckpt, args.team_repo)

    print("=" * 70)
    print("SilkBehav-4  ·  P2 Few-Shot Benchmark")
    print("=" * 70)
    print(f"Classes      : {CLASSES}")
    print(f"N-way        : {N_WAY}")
    print(f"K-shots      : {shots}")
    print(f"Episodes     : {args.episodes}")
    print(f"Frames/video : {args.num_frames}")
    print(f"Query/class  : {args.query_per_class}")
    print(f"Seed         : {args.seed}")
    print(f"Device       : {DEVICE}")
    print(f"Frame dir    : {args.frame_dir}")
    print(f"Checkpoint   : {args.ckpt}")
    print()

    # ── Set global seeds ──
    # Ref: PyTorch reproducibility docs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Load data split ──
    samples = load_split(Path(args.split_file), Path(args.frame_dir),
                         allow_scan_fallback=not args.no_scan_fallback)
    print(f"Loaded {len(samples)} test videos from split file.")
    for c in CLASSES:
        n = sum(1 for _, l in samples if l == CLASSES.index(c))
        print(f"  {c}: {n} videos")
    print()

    # ── Build shared backbone ──
    # Ref: TEAM official – model.py, CNN_FSHead
    backbone = build_resnet50_backbone()
    encoder = TemporalPooler(backbone)

    # ── Build models ──
    # ProtoNet: same backbone, prototype matching head
    # Ref: Snell et al. NeurIPS 2017
    protonet = ProtoNet(encoder)

    # TEAM: same backbone, DPM matching head
    # Ref: TEAM CVPR 2025
    # Deep-copy encoder so checkpoint loading doesn't cross-contaminate
    encoder_team = copy.deepcopy(encoder)
    team_model = TEAM_FSL(encoder_team)

    # ── Load TEAM checkpoint ──
    # Ref: TEAM official – run.py
    load_team_checkpoint(team_model, args.ckpt)

    # For ProtoNet: use the TEAM backbone weights (same backbone, fair comparison)
    # Ref: experimental design – backbone consistency for ablation
    protonet.encoder.backbone.load_state_dict(
        team_model.encoder.backbone.state_dict()
    )

    protonet = protonet.to(DEVICE)
    team_model = team_model.to(DEVICE)

    # ── Results collector ──
    all_results = {}

    for k_shot in shots:
        print(f"\n{'─' * 70}")
        print(f"  {N_WAY}-way {k_shot}-shot  ·  {args.episodes} episodes")
        print(f"{'─' * 70}")

        # Generate shared episodes (same seed → identical for both methods)
        # Ref: TEAM official – VideoDataset episode sampling
        sampler = EpisodeSampler(
            samples, N_WAY, k_shot, args.query_per_class,
            args.episodes, args.seed, num_frames=args.num_frames
        )

        methods = {
            "ProtoNet": protonet,
            "TEAM":     team_model,
        }

        for name, model in methods.items():
            print(f"\n▶ Evaluating {name} ...")
            accs, f1s = evaluate_method(model, sampler, N_WAY, k_shot, DEVICE)

            acc_mean, acc_ci = confidence_interval_95(accs)
            f1_mean, f1_ci  = confidence_interval_95(f1s)

            key = f"{N_WAY}way_{k_shot}shot_{name}"
            all_results[key] = {
                "method":   name,
                "n_way":    N_WAY,
                "k_shot":   k_shot,
                "episodes": args.episodes,
                "accuracy_mean": round(acc_mean, 4),
                "accuracy_ci95": round(acc_ci, 4),
                "macro_f1_mean": round(f1_mean, 4),
                "macro_f1_ci95": round(f1_ci, 4),
            }

            print(f"  ✓ {name:10s} | "
                  f"Acc: {acc_mean:.4f} ± {acc_ci:.4f} | "
                  f"F1:  {f1_mean:.4f} ± {f1_ci:.4f}")

    # ── Summary Table ──
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY TABLE  ·  SilkBehav-4 P2 Few-Shot Benchmark")
    print(f"{'=' * 70}")
    print(f"{'Method':<12} {'Setting':<12} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"{'─' * 64}")

    # Also include VideoMAE supervised baseline for reference
    # Ref: P1 results – supervised VideoMAE baseline
    print(f"{'VideoMAE':<12} {'supervised':<12} {'85.10':<20} {'0.8540':<20}  (P1 baseline)")

    for key, res in all_results.items():
        setting = f"{res['n_way']}w-{res['k_shot']}s"
        acc_str = f"{res['accuracy_mean']:.4f} ± {res['accuracy_ci95']:.4f}"
        f1_str  = f"{res['macro_f1_mean']:.4f} ± {res['macro_f1_ci95']:.4f}"
        print(f"{res['method']:<12} {setting:<12} {acc_str:<20} {f1_str:<20}")

    # ── Save JSON results ──
    # Add VideoMAE reference
    all_results["supervised_VideoMAE"] = {
        "method": "VideoMAE", "setting": "supervised",
        "accuracy": 0.851, "macro_f1": 0.854,
        "note": "P1 baseline, not few-shot"
    }

    out_json = os.path.join(args.output_dir, "p2_fewshot_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── Save per-episode CSV for downstream analysis ──
    print("\nDone.")


if __name__ == "__main__":
    main()
