#!/usr/bin/env python3
"""
SilkBehav-4 Supervised Baseline: 3D-ResNet50
=============================================

目的：在 SilkBehav-4 数据集上跑 supervised 4-class classification baseline
      产出 accuracy / Macro-F1 / per-class F1 / confusion matrix
      这是论文 Table 1 的核心数字之一

用法：
    # 单卡训练
    python scripts/exp001_supervised_train.py --gpu 0

    # 指定 epoch 数
    python scripts/exp001_supervised_train.py --gpu 0 --epochs 50

输出：
    experiments/exp001_supervised_baseline/
    ├── best_model.pth
    ├── training_log.csv
    ├── test_results.json
    ├── confusion_matrix.png
    └── classification_report.txt
"""

import os
import sys
import json
import csv
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# =====================================================
# 配置
# =====================================================
PROJECT_ROOT = "/home/fmh/SilkBehav-4"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
SPLIT_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "silkbehav4_valid1153.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "exp001_supervised_baseline")

CLASSES = ["feeding", "head_swing", "inactive", "locomotion"]  # 字母序，跟 label_id 对应
NUM_CLASSES = 4
NUM_FRAMES = 16        # 每个 clip 采样帧数
CLIP_LEN_SEC = 5       # 每个 clip 约 5 秒
FRAME_SIZE = 112       # 3D-ResNet 标准输入
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 数据集
# =====================================================
class SilkBehavDataset(Dataset):
    """从按类别文件夹组织的视频 clips 加载数据"""
    
    def __init__(self, samples, num_frames=16, frame_size=112, augment=False):
        """
        Args:
            samples: list of (clip_path, label_id)
            num_frames: 每个 clip 均匀采样的帧数
            frame_size: 输出帧的空间分辨率
            augment: 是否做数据增强
        """
        self.samples = samples
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path):
        """从视频文件中均匀采样 num_frames 帧"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"视频帧数为 0: {video_path}")
        
        # 均匀采样
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # 帧数不够就重复最后一帧
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)
            indices = np.array(indices[:self.num_frames])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # 读取失败用黑帧填充
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _transform_frames(self, frames):
        """把帧列表转成 tensor [C, T, H, W]"""
        import cv2
        
        processed = []
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            
            if self.augment:
                # 随机水平翻转
                if random.random() > 0.5:
                    frame = np.fliplr(frame).copy()
                # 随机亮度调整
                delta = random.uniform(-20, 20)
                frame = np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            
            # Normalize to [0, 1] then standardize
            frame = frame.astype(np.float32) / 255.0
            # ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            
            processed.append(frame)
        
        # [T, H, W, C] -> [C, T, H, W]
        video = np.stack(processed, axis=0)  # [T, H, W, C]
        video = video.transpose(3, 0, 1, 2)  # [C, T, H, W]
        
        return torch.FloatTensor(video)
    
    def __getitem__(self, idx):
        clip_path, label_id = self.samples[idx]
        
        try:
            frames = self._load_video_frames(clip_path)
            video = self._transform_frames(frames)
        except Exception as e:
            print(f"⚠️ 加载失败 {clip_path}: {e}", file=sys.stderr)
            # 返回零张量
            video = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        
        return video, label_id


def load_split_samples(split_csv, split_name):
    """从 CSV 加载指定 split 的样本"""
    samples = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split_name:
                clip_path = row["clip_path"]
                label_id = int(row["label_id"])
                if os.path.isfile(clip_path):
                    samples.append((clip_path, label_id))
                else:
                    print(f"⚠️ 文件不存在: {clip_path}", file=sys.stderr)
    return samples


# =====================================================
# 模型：3D-ResNet50
# =====================================================
def build_model(num_classes, pretrained=True):
    """构建 3D-ResNet50，使用 Kinetics400 预训练权重"""
    try:
        # PyTorch >= 2.0 的方式
        from torchvision.models.video import r3d_18, R3D_18_Weights
        
        if pretrained:
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            print("✅ 加载 R3D-18 Kinetics400 预训练权重")
        else:
            model = r3d_18(weights=None)
            print("⚠️ 无预训练权重")
        
        # 替换最后的 FC 层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"   FC 层: {in_features} -> {num_classes}")
        
    except ImportError:
        # 旧版 torchvision
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model


# =====================================================
# 训练
# =====================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * videos.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}, "
                  f"loss={loss.item():.4f}, acc={correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        
        outputs = model(videos)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * videos.size(0)
        
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
    accuracy = (all_preds == all_labels).mean()
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_metrics(y_true, y_pred, classes):
    """计算 per-class F1, Macro-F1, confusion matrix"""
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        f1_score, accuracy_score
    )
    
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": {cls: float(f) for cls, f in zip(classes, per_class_f1)},
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def plot_confusion_matrix(cm, classes, save_path):
    """保存 confusion matrix 图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Supervised Baseline - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"   混淆矩阵已保存: {save_path}")
    except Exception as e:
        print(f"   ⚠️ 绘图失败: {e}")


# =====================================================
# 主函数
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="SilkBehav-4 Supervised Baseline")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SilkBehav-4 Supervised Baseline")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Frames per clip: {args.num_frames}")
    print(f"Frame size: {args.frame_size}")
    print(f"Seed: {args.seed}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print(f"\n📖 加载数据...")
    train_samples = load_split_samples(SPLIT_CSV, "train")
    val_samples = load_split_samples(SPLIT_CSV, "val")
    test_samples = load_split_samples(SPLIT_CSV, "test")
    
    print(f"   Train: {len(train_samples)}")
    print(f"   Val:   {len(val_samples)}")
    print(f"   Test:  {len(test_samples)}")
    
    if len(train_samples) == 0:
        print("❌ 训练集为空！检查 SPLIT_CSV 路径和 split 字段")
        sys.exit(1)
    
    # 类别分布
    train_labels = [s[1] for s in train_samples]
    print(f"   Train 类别分布: {Counter(train_labels)}")
    
    # 构建 DataLoader
    train_dataset = SilkBehavDataset(
        train_samples, num_frames=args.num_frames, 
        frame_size=args.frame_size, augment=True
    )
    val_dataset = SilkBehavDataset(
        val_samples, num_frames=args.num_frames, 
        frame_size=args.frame_size, augment=False
    )
    test_dataset = SilkBehavDataset(
        test_samples, num_frames=args.num_frames, 
        frame_size=args.frame_size, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 构建模型
    print(f"\n🏗️ 构建模型...")
    model = build_model(NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # 类别权重（处理不平衡）
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1)) 
        for i in range(NUM_CLASSES)
    ]).to(device)
    print(f"   类别权重: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    print(f"\n🚀 开始训练...")
    best_val_f1 = 0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, macro-F1={val_f1:.4f}")
        print(f"  LR:    {current_lr:.6f}")
        
        # 记录日志
        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(current_lr, 8),
        })
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, best_path)
            print(f"  ✅ 新 best! Macro-F1={val_f1:.4f}, 已保存 {best_path}")
    
    # 保存训练日志
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    print(f"\n📊 训练日志已保存: {log_path}")
    
    # 加载最佳模型，在 test set 上评估
    print(f"\n{'='*60}")
    print(f"在 Test Set 上评估 (best model, epoch with val-F1={best_val_f1:.4f})")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_preds, CLASSES)
    
    print(f"\n📋 Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"\n   Per-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        print(f"     {cls}: {f1:.4f}")
    print(f"\n{metrics['report']}")
    
    # 保存 test 结果
    results = {
        "model": "R3D-18 (Kinetics400 pretrained)",
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": float(best_val_f1),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "test_confusion_matrix": metrics["confusion_matrix"],
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_frames": args.num_frames,
            "frame_size": args.frame_size,
            "seed": args.seed,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Test 结果已保存: {results_path}")
    
    # 保存 classification report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"SilkBehav-4 Supervised Baseline Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: R3D-18 (Kinetics400 pretrained)\n")
        f.write(f"Best epoch: {checkpoint['epoch']}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Macro-F1: {metrics['macro_f1']:.4f}\n\n")
        f.write(metrics["report"])
    print(f"📄 Report 已保存: {report_path}")
    
    # 绘制 confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), 
        CLASSES, cm_path
    )
    
    print(f"\n{'='*60}")
    print(f"✅ 实验完成！")
    print(f"{'='*60}")
    print(f"   Best Val Macro-F1:  {best_val_f1:.4f}")
    print(f"   Test Accuracy:      {metrics['accuracy']:.4f}")
    print(f"   Test Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"   输出目录:           {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
