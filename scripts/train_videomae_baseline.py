#!/usr/bin/env python3
"""
SilkBehav-4 · VideoMAE Supervised Baseline
==========================================
decord 读视频 + HuggingFace Trainer，完全离线运行。

用法:
    cd /home/fmh/SilkBehav-4
    conda activate silkbehav
    CUDA_VISIBLE_DEVICES=0 python scripts/train_videomae_baseline.py

参考来源:
- HuggingFace VideoMAE 文档: https://huggingface.co/docs/transformers/model_doc/videomae
- HuggingFace Video Classification 教程: https://huggingface.co/docs/transformers/tasks/video_classification
- decord 官方示例: https://github.com/dmlc/decord#usage
- VideoMAE 论文: Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners
  for Self-Supervised Video Pre-Training", NeurIPS 2022
"""

import os
import sys
import csv
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# --- decord 配置 ---
# 参考: https://github.com/dmlc/decord#usage
# 必须在 import decord 后立即设置 bridge，让 decord 直接输出 torch.Tensor
from decord import VideoReader, cpu
import decord
decord.bridge.set_bridge("torch")  # 直接输出 torch.Tensor，避免 numpy 中转

# --- HuggingFace ---
# 参考: https://huggingface.co/docs/transformers/model_doc/videomae
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)

# --- metrics ---
# 参考: https://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 配置 —— 按需修改这里
# ============================================================
# 参考: HuggingFace TrainingArguments 文档
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

class Config:
    """集中管理所有路径和超参数，方便修改。"""

    # --- 路径 ---
    project_root = Path("/home/fmh/SilkBehav-4")
    csv_path = project_root / "data" / "splits" / "silkbehav4_valid1153.csv"
    video_root = project_root / "data" / "raw_videos"
    model_path = project_root / "models" / "pretrained" / "videomae-base-finetuned-kinetics"
    output_dir = project_root / "models" / "videomae_baseline_run1"

    # --- 类别映射 ---
    # 与文件夹名一致，按字母序编号
    label2id = {
        "feeding": 0,
        "head_swing": 1,
        "inactive": 2,
        "locomotion": 3,
    }
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    # --- 视频采样 ---
    # 参考: VideoMAE 默认输入为 16 帧，224×224
    # https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics
    num_frames = 16  # VideoMAE 默认
    clip_duration_sec = None  # None = 用整段视频均匀采样 16 帧

    # --- 训练超参 ---
    num_epochs = 30
    batch_size = 4  # 单卡 5090 (32GB)，VideoMAE-base 约 6-8GB 显存
    learning_rate = 5e-5
    weight_decay = 0.05
    warmup_ratio = 0.1
    eval_strategy = "epoch"
    save_strategy = "epoch"
    save_total_limit = 3
    load_best_model_at_end = True
    metric_for_best_model = "f1_macro"
    seed = 42

    # --- 数据划分 ---
    # CSV 中如果没有 split 列，按 80/20 随机划分
    val_ratio = 0.2


# ============================================================
# 2. CSV 解析
# ============================================================
# 你的 CSV 格式预期:
#   video_path, label  (最少两列)
#   或: video_path, label, split  (三列，split 为 train/val)
#
# video_path 可以是:
#   - 绝对路径: /home/fmh/SilkBehav-4/data/raw_videos/feeding/clip_001.mp4
#   - 相对路径: feeding/clip_001.mp4  (相对于 video_root)
#   - 纯文件名: clip_001.mp4  (会在所有子目录中搜索)

def parse_csv(cfg: Config) -> Tuple[List[Dict], List[Dict]]:
    """
    解析 CSV，返回 (train_samples, val_samples)。
    每个 sample = {"video_path": str, "label": int}
    """
    logger.info(f"读取 CSV: {cfg.csv_path}")
    samples = []

    with open(cfg.csv_path, "r", encoding="utf-8") as f:
        # 自动检测分隔符
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)

        # 自动检测列名（兼容多种命名）
        fieldnames = reader.fieldnames
        logger.info(f"CSV 列名: {fieldnames}")

        # 寻找 video 路径列
        path_col = None
        for candidate in ["video_path", "path", "file", "filename", "video", "clip_path"]:
            if candidate in fieldnames:
                path_col = candidate
                break
        if path_col is None:
            # 如果找不到，用第一列
            path_col = fieldnames[0]
            logger.warning(f"未找到已知路径列名，使用第一列: '{path_col}'")

        # 寻找 label 列
        label_col = None
        for candidate in ["label", "class", "category", "behavior", "action"]:
            if candidate in fieldnames:
                label_col = candidate
                break
        if label_col is None:
            label_col = fieldnames[1]
            logger.warning(f"未找到已知标签列名，使用第二列: '{label_col}'")

        # 寻找 split 列
        split_col = None
        for candidate in ["split", "subset", "set", "fold"]:
            if candidate in fieldnames:
                split_col = candidate
                break

        for row in reader:
            raw_path = row[path_col].strip()
            label_str = row[label_col].strip().lower()

            # 解析标签
            if label_str in cfg.label2id:
                label_id = cfg.label2id[label_str]
            elif label_str.isdigit():
                label_id = int(label_str)
            else:
                logger.warning(f"未知标签 '{label_str}'，跳过: {raw_path}")
                continue

            # 解析视频路径
            video_path = _resolve_video_path(raw_path, cfg.video_root)
            if video_path is None:
                logger.warning(f"视频文件不存在，跳过: {raw_path}")
                continue

            split_val = row[split_col].strip().lower() if split_col else None
            samples.append({
                "video_path": str(video_path),
                "label": label_id,
                "split": split_val,
            })

    logger.info(f"共解析 {len(samples)} 条有效记录")

    # 划分 train / val
    if samples and samples[0]["split"] is not None:
        train = [s for s in samples if s["split"] in ("train", "training")]
        val = [s for s in samples if s["split"] in ("val", "valid", "validation", "test")]
        logger.info(f"CSV 内置 split: train={len(train)}, val={len(val)}")
    else:
        random.seed(cfg.seed)
        random.shuffle(samples)
        n_val = int(len(samples) * cfg.val_ratio)
        val = samples[:n_val]
        train = samples[n_val:]
        logger.info(f"随机划分 ({1-cfg.val_ratio:.0%}/{cfg.val_ratio:.0%}): train={len(train)}, val={len(val)}")

    return train, val


def _resolve_video_path(raw_path: str, video_root: Path) -> Optional[Path]:
    """尝试多种方式解析视频路径。"""
    # 1. 绝对路径
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    # 2. 相对于 video_root
    p = video_root / raw_path
    if p.exists():
        return p

    # 3. 如果是纯文件名，在子目录中搜索
    if "/" not in raw_path and "\\" not in raw_path:
        for subdir in video_root.iterdir():
            if subdir.is_dir():
                candidate = subdir / raw_path
                if candidate.exists():
                    return candidate

    # 4. 尝试 follow symlinks（readlink）
    p = video_root / raw_path
    try:
        resolved = p.resolve()
        if resolved.exists():
            return resolved
    except (OSError, RuntimeError):
        pass

    return None


# ============================================================
# 3. 视频采样 —— 用 decord
# ============================================================
# 参考: https://github.com/dmlc/decord#usage
# 参考: HuggingFace Video Classification 教程中的 sample_frame_indices
# https://huggingface.co/docs/transformers/tasks/video_classification

def sample_frame_indices(total_frames: int, num_frames: int = 16) -> np.ndarray:
    """
    从视频中均匀采样 num_frames 个帧索引。
    参考: HuggingFace 官方教程 sample_frame_indices 函数
    """
    if total_frames >= num_frames:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    else:
        # 帧数不够，重复最后一帧
        indices = np.arange(total_frames, dtype=int)
        pad = np.full(num_frames - total_frames, total_frames - 1, dtype=int)
        indices = np.concatenate([indices, pad])
    return indices


def read_video_decord(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """
    用 decord 读取视频并均匀采样指定帧数。

    参考: https://github.com/dmlc/decord#usage
    返回: torch.Tensor, shape = (num_frames, H, W, C), dtype=uint8

    decord.bridge 已设为 "torch"，所以 vr.get_batch() 直接返回 torch.Tensor。
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"视频帧数为 0: {video_path}")

    indices = sample_frame_indices(total_frames, num_frames)
    frames = vr.get_batch(indices)  # (num_frames, H, W, C), torch.uint8

    return frames


# ============================================================
# 4. Dataset
# ============================================================
# 参考: https://huggingface.co/docs/transformers/tasks/video_classification
# 参考: VideoMAEImageProcessor 文档
# https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEImageProcessor

class SilkBehavDataset(Dataset):
    """
    PyTorch Dataset，用 decord 读视频，用 VideoMAEImageProcessor 做预处理。

    HuggingFace 官方教程要求 pixel_values shape = (num_frames, C, H, W)。
    VideoMAEImageProcessor 接受 list of numpy arrays (num_frames, H, W, C)，
    输出 pixel_values shape = (num_frames, C, H, W)，已归一化、resize、center crop。
    """

    def __init__(
        self,
        samples: List[Dict],
        processor: VideoMAEImageProcessor,
        num_frames: int = 16,
    ):
        self.samples = samples
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        try:
            # decord 读视频 → (num_frames, H, W, C), torch.uint8
            frames = read_video_decord(video_path, self.num_frames)

            # VideoMAEImageProcessor 需要 list of numpy arrays
            # 每个 array shape = (H, W, C), dtype=uint8
            # 参考: https://huggingface.co/docs/transformers/model_doc/videomae
            frames_list = [f.numpy() for f in frames]

            # 预处理: resize 224×224, center crop, normalize
            inputs = self.processor(frames_list, return_tensors="pt")

            # inputs["pixel_values"] shape = (1, num_frames, C, H, W)
            # Trainer 需要去掉 batch 维度
            pixel_values = inputs["pixel_values"].squeeze(0)

        except Exception as e:
            logger.error(f"读取视频失败 [{video_path}]: {e}")
            # 返回全零 tensor，训练时不会崩，但会产生噪声
            # 更好的做法是在 DataLoader 层面跳过，但 Trainer 不方便做
            pixel_values = torch.zeros(
                self.num_frames, 3, 224, 224, dtype=torch.float32
            )

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 5. Metrics
# ============================================================
# 参考: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_metrics

def compute_metrics(eval_pred):
    """
    Trainer 的 compute_metrics 回调。
    参考: HuggingFace Trainer 文档
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# 6. 自定义 Trainer（可选：打印 classification report）
# ============================================================
# 参考: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer

class VideoMAETrainer(Trainer):
    """继承 Trainer，在 evaluate 结束后打印详细分类报告。"""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 打印详细报告（仅在主进程）
        if self.is_world_process_zero():
            eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            preds_output = self.predict(eval_ds)
            preds = np.argmax(preds_output.predictions, axis=-1)
            labels = preds_output.label_ids

            report = classification_report(
                labels, preds,
                target_names=list(Config.label2id.keys()),
                digits=4,
            )
            logger.info(f"\n{'='*60}\nClassification Report:\n{report}{'='*60}")

        return output


# ============================================================
# 7. 主函数
# ============================================================

def main():
    cfg = Config()

    # --- 检查路径 ---
    assert cfg.csv_path.exists(), f"CSV 不存在: {cfg.csv_path}"
    assert cfg.video_root.exists(), f"视频目录不存在: {cfg.video_root}"
    assert cfg.model_path.exists(), f"模型权重不存在: {cfg.model_path}"

    logger.info(f"项目根目录: {cfg.project_root}")
    logger.info(f"模型路径: {cfg.model_path}")
    logger.info(f"类别: {cfg.label2id}")

    # --- 解析数据 ---
    train_samples, val_samples = parse_csv(cfg)

    # 统计各类别数量
    for split_name, split_data in [("Train", train_samples), ("Val", val_samples)]:
        counts = {}
        for s in split_data:
            lbl = cfg.id2label[s["label"]]
            counts[lbl] = counts.get(lbl, 0) + 1
        logger.info(f"{split_name} 分布: {counts}")

    # --- 加载 processor 和 model ---
    # 参考: https://huggingface.co/docs/transformers/model_doc/videomae
    # 完全离线加载，不访问网络
    logger.info("加载 VideoMAEImageProcessor（离线）...")
    processor = VideoMAEImageProcessor.from_pretrained(
        str(cfg.model_path),
        local_files_only=True,
    )

    logger.info("加载 VideoMAEForVideoClassification（离线）...")
    model = VideoMAEForVideoClassification.from_pretrained(
        str(cfg.model_path),
        num_labels=cfg.num_labels,
        label2id=cfg.label2id,
        id2label=cfg.id2label,
        ignore_mismatched_sizes=True,  # Kinetics-400 头有 400 类，我们只有 4 类，需要忽略尺寸不匹配
        local_files_only=True,
    )
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 构建 Dataset ---
    train_dataset = SilkBehavDataset(train_samples, processor, cfg.num_frames)
    val_dataset = SilkBehavDataset(val_samples, processor, cfg.num_frames)

    # --- 验证一个样本 ---
    logger.info("验证第一个训练样本...")
    sample = train_dataset[0]
    logger.info(f"  pixel_values shape: {sample['pixel_values'].shape}")  # 期望 (16, 3, 224, 224)
    logger.info(f"  label: {sample['labels'].item()} ({cfg.id2label[sample['labels'].item()]})")

    # --- 训练参数 ---
    # 参考: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,  # eval 不需要梯度，可以大一些
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        logging_steps=10,
        logging_first_step=True,
        remove_unused_columns=False,  # 重要！Dataset 返回自定义 dict，不能让 Trainer 自动删列
        fp16=True,  # 5090 支持，节省显存加速训练
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=cfg.seed,
        report_to="none",  # 离线，不上传到 wandb 等
    )

    # --- Trainer ---
    trainer = VideoMAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 训练 ---
    logger.info("开始训练...")
    train_result = trainer.train()

    # --- 保存 ---
    logger.info("保存最佳模型...")
    trainer.save_model(str(cfg.output_dir / "best_model"))
    processor.save_pretrained(str(cfg.output_dir / "best_model"))

    # --- 最终评估 ---
    logger.info("最终评估...")
    metrics = trainer.evaluate()
    logger.info(f"最终指标: {metrics}")

    # --- 保存训练指标 ---
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    logger.info(f"完成！模型保存在: {cfg.output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
