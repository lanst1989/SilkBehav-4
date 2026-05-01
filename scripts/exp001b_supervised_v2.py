#!/usr/bin/env python3
"""
SilkBehav-4 Supervised Baseline v2 (修正版)
============================================

修正内容（相比 exp001）：
1. LR: 1e-3 -> 1e-4（预训练模型 fine-tune 标准）
2. 冻结 stem + layer1 + layer2，只 fine-tune layer3 + layer4 + fc
3. 帧数: 16 -> 32（蚕运动慢，需要更多时序信息）
4. 增加 warmup（前 5 epoch LR 线性增长）
5. Label smoothing 0.1（防止过拟合）

用法：
    python scripts/exp001b_supervised_v2.py --gpu 0 --epochs 60
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

# =====================================================
# 配置
# =====================================================
PROJECT_ROOT = "/home/fmh/SilkBehav-4"
SPLIT_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "silkbehav4_valid1153.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "exp001b_supervised_v2")

CLASSES = ["feeding", "head_swing", "inactive", "locomotion"]
NUM_CLASSES = 4
NUM_FRAMES = 32         # 增加到 32 帧
FRAME_SIZE = 128        # 稍大分辨率
BATCH_SIZE = 8          # 32帧×128px 显存占用更大，降 batch
NUM_WORKERS = 4
LR = 1e-4               # 降低 10 倍
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 数据集（与 exp001 相同，但支持更多帧）
# =====================================================
class SilkBehavDataset(Dataset):
    def __init__(self, samples, num_frames=32, frame_size=128, augment=False):
        self.samples = samples
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"视频帧数为 0: {video_path}")
        
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)
            indices = np.array(indices[:self.num_frames])
        
        # 训练时加随机时序抖动
        if self.augment and total_frames >= self.num_frames:
            jitter = np.random.randint(-2, 3, size=len(indices))
            indices = np.clip(indices + jitter, 0, total_frames - 1)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _transform_frames(self, frames):
        import cv2
        processed = []
        
        # 训练时：随机 crop + 翻转
        if self.augment:
            crop_size = self.frame_size
            # 先 resize 到稍大尺寸，再随机 crop
            resize_to = int(self.frame_size * 1.15)
            crop_x = random.randint(0, resize_to - crop_size)
            crop_y = random.randint(0, resize_to - crop_size)
            do_flip = random.random() > 0.5
            brightness_delta = random.uniform(-15, 15)
        else:
            resize_to = self.frame_size
            crop_x = crop_y = 0
            crop_size = self.frame_size
            do_flip = False
            brightness_delta = 0
        
        for frame in frames:
            frame = cv2.resize(frame, (resize_to, resize_to))
            
            if self.augment:
                frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                if do_flip:
                    frame = np.fliplr(frame).copy()
                frame = np.clip(frame.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
            
            frame = frame.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            processed.append(frame)
        
        video = np.stack(processed, axis=0)     # [T, H, W, C]
        video = video.transpose(3, 0, 1, 2)     # [C, T, H, W]
        return torch.FloatTensor(video)
    
    def __getitem__(self, idx):
        clip_path, label_id = self.samples[idx]
        try:
            frames = self._load_video_frames(clip_path)
            video = self._transform_frames(frames)
        except Exception as e:
            print(f"⚠️ 加载失败 {clip_path}: {e}", file=sys.stderr)
            video = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        return video, label_id


def load_split_samples(split_csv, split_name):
    samples = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split_name:
                clip_path = row["clip_path"]
                label_id = int(row["label_id"])
                if os.path.isfile(clip_path):
                    samples.append((clip_path, label_id))
    return samples


# =====================================================
# 模型：R3D-18 + 部分冻结
# =====================================================
def build_model(num_classes):
    """R3D-18 + 冻结前半部分"""
    from torchvision.models.video import r3d_18, R3D_18_Weights
    
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    print("✅ 加载 R3D-18 Kinetics400 预训练权重")
    
    # 冻结 stem + layer1 + layer2
    frozen_parts = ["stem", "layer1", "layer2"]
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(part) for part in frozen_parts):
            param.requires_grad = False
            frozen_count += 1
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   冻结: {frozen_parts}")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 替换 FC
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    print(f"   FC: {in_features} -> Dropout(0.3) -> {num_classes}")
    
    return model


# =====================================================
# 训练
# =====================================================
def get_lr(epoch, warmup_epochs, base_lr, total_epochs):
    """Warmup + Cosine Annealing"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        import math
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item() * videos.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
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
        ax.set_title("Supervised Baseline v2 - Confusion Matrix")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"SilkBehav-4 Supervised Baseline v2")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr} (with {WARMUP_EPOCHS}-epoch warmup)")
    print(f"Frames: {NUM_FRAMES}, Size: {FRAME_SIZE}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Frozen layers: stem + layer1 + layer2")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 数据
    print(f"\n📖 加载数据...")
    train_samples = load_split_samples(SPLIT_CSV, "train")
    val_samples = load_split_samples(SPLIT_CSV, "val")
    test_samples = load_split_samples(SPLIT_CSV, "test")
    
    print(f"   Train: {len(train_samples)}")
    print(f"   Val:   {len(val_samples)}")
    print(f"   Test:  {len(test_samples)}")
    
    train_labels = [s[1] for s in train_samples]
    print(f"   Train 类别分布: {Counter(train_labels)}")
    
    train_loader = DataLoader(
        SilkBehavDataset(train_samples, NUM_FRAMES, FRAME_SIZE, augment=True),
        batch_size=args.batch_size, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        SilkBehavDataset(val_samples, NUM_FRAMES, FRAME_SIZE, augment=False),
        batch_size=args.batch_size, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        SilkBehavDataset(test_samples, NUM_FRAMES, FRAME_SIZE, augment=False),
        batch_size=args.batch_size, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 模型
    print(f"\n🏗️ 构建模型...")
    model = build_model(NUM_CLASSES)
    model = model.to(device)
    
    # 类别权重
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1)) 
        for i in range(NUM_CLASSES)
    ]).to(device)
    print(f"   类别权重: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    # 训练
    print(f"\n🚀 开始训练...")
    best_val_f1 = 0
    training_log = []
    patience = 0
    max_patience = 15  # early stopping
    
    for epoch in range(args.epochs):
        # 手动调 LR (warmup + cosine)
        current_lr = get_lr(epoch, WARMUP_EPOCHS, args.lr, args.epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}) ---")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, macro-F1={val_f1:.4f}")
        
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(current_lr, 8),
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ✅ 新 best! Macro-F1={val_f1:.4f}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} (patience={max_patience})")
                break
    
    # 保存日志
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    
    # Test 评估
    print(f"\n{'='*60}")
    print(f"Test Set 评估 (best val-F1={best_val_f1:.4f})")
    print(f"{'='*60}")
    
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_preds, CLASSES)
    
    print(f"\n📋 Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Macro-F1:  {metrics['macro_f1']:.4f}")
    for cls, f1 in metrics["per_class_f1"].items():
        print(f"     {cls}: {f1:.4f}")
    print(f"\n{metrics['report']}")
    
    # 保存结果
    results = {
        "model": "R3D-18 (Kinetics400 pretrained, frozen stem+layer1+layer2)",
        "best_epoch": ckpt["epoch"],
        "best_val_f1": float(best_val_f1),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "test_confusion_matrix": metrics["confusion_matrix"],
        "config": {
            "epochs_run": len(training_log),
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_epochs": WARMUP_EPOCHS,
            "label_smoothing": LABEL_SMOOTHING,
            "num_frames": NUM_FRAMES,
            "frame_size": FRAME_SIZE,
            "frozen_layers": "stem+layer1+layer2",
            "seed": args.seed,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"SilkBehav-4 Supervised Baseline v2\n{'='*50}\n")
        f.write(f"Model: R3D-18 frozen(stem+L1+L2)\n")
        f.write(f"Best epoch: {ckpt['epoch']}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Macro-F1: {metrics['macro_f1']:.4f}\n\n")
        f.write(metrics["report"])
    
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        CLASSES,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    
    print(f"\n{'='*60}")
    print(f"✅ 实验完成！")
    print(f"   Best Val Macro-F1:  {best_val_f1:.4f}")
    print(f"   Test Accuracy:      {metrics['accuracy']:.4f}")
    print(f"   Test Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"   输出目录:           {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
