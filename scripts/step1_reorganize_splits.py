#!/usr/bin/env python3
"""
SilkBehav-4 数据目录重组（适配 pytorchvideo UCF101 格式）

参考：HuggingFace 官方 VideoMAE fine-tune 教程
https://huggingface.co/docs/transformers/tasks/video_classification

官方教程要求的目录结构：
    train/
        ClassName1/
            video_1.mp4
        ClassName2/
            video_1.mp4
    val/
        ClassName1/
            ...
    test/
        ClassName1/
            ...

本脚本从 silkbehav4_valid1153.csv 读取 split 信息，
用符号链接（不复制）把视频按 split x class 组织好。
"""

import os
import csv

SPLIT_CSV = "/home/fmh/SilkBehav-4/data/splits/silkbehav4_valid1153.csv"
DST_ROOT = "/home/fmh/SilkBehav-4/data/videomae_splits"
CLASSES = ["feeding", "head_swing", "inactive", "locomotion"]
SPLITS = ["train", "val", "test"]


def main():
    # 1. 创建目录结构
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)
    
    # 2. 读取 CSV，建立符号链接
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    errors = 0
    
    with open(SPLIT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            label = row["label_main"]
            clip_path = row["clip_path"]
            sample_id = row["sample_id"]
            
            if split not in SPLITS or label not in CLASSES:
                continue
            
            if not os.path.isfile(clip_path):
                print(f"⚠️ 文件不存在: {clip_path}")
                errors += 1
                continue
            
            # 用 sample_id 作为文件名（保证唯一）
            safe_name = sample_id.replace("/", "_").replace("\\", "_")
            if not safe_name.endswith(".mp4"):
                safe_name += ".mp4"
            
            link_path = os.path.join(DST_ROOT, split, label, safe_name)
            
            if not os.path.exists(link_path):
                os.symlink(clip_path, link_path)
            
            counts[split][label] += 1
    
    # 3. 打印结果
    print("=" * 50)
    print("SilkBehav-4 数据重组完成")
    print(f"目标目录: {DST_ROOT}")
    print("=" * 50)
    
    total = 0
    for split in SPLITS:
        split_total = sum(counts[split].values())
        total += split_total
        print(f"\n{split}/ ({split_total} videos):")
        for cls in CLASSES:
            print(f"  {cls}/: {counts[split][cls]}")
    
    print(f"\n总计: {total}")
    if errors:
        print(f"⚠️ {errors} 个文件未找到")
    else:
        print("✅ 0 错误")


if __name__ == "__main__":
    main()
