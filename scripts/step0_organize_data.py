#!/usr/bin/env python3
"""
SilkBehav-4 数据整理脚本

功能：
1. 读取 valid1153 的 CSV
2. 按 label_main 把 clips 复制到 SilkBehav-4/data/raw_videos/{类别}/
3. 同时生成主线标签文件到 SilkBehav-4/data/splits/
4. 处理 .mp4.mp4 文件名问题

用法：
    python step0_organize_data.py
"""

import os
import csv
import shutil
from pathlib import Path
from collections import Counter

# =====================================================
# 路径配置
# =====================================================
VALID1153_CSV = "/home/fmh/SilkVIM/outputs/superanimal_valid_subset/ucf_few_shot_superanimal_valid.csv"
MAINLINE_CSV = "/home/fmh/SilkVIM/data/processed/splits/stage5_mainline_1330_split_with_clips.csv"

DST_ROOT = "/home/fmh/SilkBehav-4"
DST_VIDEOS = os.path.join(DST_ROOT, "data", "raw_videos")
DST_SPLITS = os.path.join(DST_ROOT, "data", "splits")

VALID_LABELS = {"feeding", "head_swing", "locomotion", "inactive"}


def find_clip_file(clip_path):
    """尝试找到实际存在的 clip 文件，处理 .mp4.mp4 问题"""
    # 直接检查
    if os.path.isfile(clip_path):
        return clip_path
    
    # 尝试加 .mp4 后缀（处理 .mp4.mp4 的情况）
    alt = clip_path + ".mp4"
    if os.path.isfile(alt):
        return alt
    
    # 尝试去掉一层 .mp4
    if clip_path.endswith(".mp4.mp4"):
        alt2 = clip_path[:-4]  # 去掉最后的 .mp4
        if os.path.isfile(alt2):
            return alt2
    
    return None


def main():
    print("=" * 60)
    print("SilkBehav-4 数据整理")
    print("=" * 60)
    
    # 1. 创建目标目录
    for label in VALID_LABELS:
        os.makedirs(os.path.join(DST_VIDEOS, label), exist_ok=True)
    os.makedirs(DST_SPLITS, exist_ok=True)
    print(f"\n✅ 目标目录已创建: {DST_VIDEOS}")
    
    # 2. 读取 valid1153
    print(f"\n📖 读取: {VALID1153_CSV}")
    rows = []
    with open(VALID1153_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"   共 {len(rows)} 条记录")
    
    # 3. 统计类别分布
    label_counts = Counter(row["label_main"] for row in rows)
    print(f"\n📊 类别分布:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    
    # 4. 检查是否有意外类别
    unexpected = set(label_counts.keys()) - VALID_LABELS
    if unexpected:
        print(f"\n⚠️  发现意外类别: {unexpected}")
        print("   这些样本将被跳过")
    
    # 5. 复制文件 + 生成新的标签表
    new_manifest = []
    success = 0
    fail = 0
    fail_list = []
    
    for row in rows:
        label = row["label_main"]
        if label not in VALID_LABELS:
            continue
        
        sample_id = row["sample_id"]
        
        # 尝试多个路径字段
        clip_path = row.get("clip_path", "") or row.get("video_path", "") or row.get("file_path", "")
        
        actual_file = find_clip_file(clip_path)
        
        if actual_file is None:
            fail += 1
            fail_list.append((sample_id, clip_path))
            continue
        
        # 目标文件名：sample_id 作为文件名（清理特殊字符）
        safe_name = sample_id.replace("/", "_").replace("\\", "_")
        if not safe_name.endswith(".mp4"):
            safe_name += ".mp4"
        
        dst_path = os.path.join(DST_VIDEOS, label, safe_name)
        
        # 复制文件
        if not os.path.exists(dst_path):
            shutil.copy2(actual_file, dst_path)
        
        # 记录新 manifest
        new_manifest.append({
            "sample_id": sample_id,
            "label_main": label,
            "label_id": list(sorted(VALID_LABELS)).index(label),
            "clip_path": dst_path,
            "original_path": actual_file,
            "num_frames": row.get("num_frames", ""),
            "split": row.get("split", ""),
        })
        
        success += 1
    
    print(f"\n📦 复制结果:")
    print(f"   ✅ 成功: {success}")
    print(f"   ❌ 失败: {fail}")
    
    if fail_list:
        print(f"\n   失败详情 (前10个):")
        for sid, cp in fail_list[:10]:
            print(f"     {sid} -> {cp}")
    
    # 6. 保存新的 manifest
    manifest_path = os.path.join(DST_SPLITS, "silkbehav4_valid1153.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "label_main", "label_id", "clip_path", 
            "original_path", "num_frames", "split"
        ])
        writer.writeheader()
        writer.writerows(new_manifest)
    print(f"\n📄 Manifest 已保存: {manifest_path}")
    
    # 7. 统计最终结果
    final_counts = Counter(r["label_main"] for r in new_manifest)
    split_counts = Counter(r["split"] for r in new_manifest)
    
    print(f"\n{'=' * 60}")
    print(f"最终统计")
    print(f"{'=' * 60}")
    print(f"\n类别分布:")
    label_id_map = {}
    for r in new_manifest:
        label_id_map[r["label_main"]] = r["label_id"]
    for label in sorted(VALID_LABELS):
        cnt = final_counts.get(label, 0)
        lid = label_id_map.get(label, "?")
        print(f"  [{lid}] {label}: {cnt}")
    print(f"  总计: {sum(final_counts.values())}")
    
    print(f"\nSplit 分布:")
    for split, cnt in sorted(split_counts.items()):
        print(f"  {split}: {cnt}")
    
    # 8. 验证文件数量
    print(f"\n磁盘验证:")
    for label in sorted(VALID_LABELS):
        label_dir = os.path.join(DST_VIDEOS, label)
        n_files = len([f for f in os.listdir(label_dir) if f.endswith(".mp4")])
        print(f"  {label}/: {n_files} files")
    
    print(f"\n✅ 整理完成！")
    print(f"   数据目录: {DST_VIDEOS}")
    print(f"   标签文件: {manifest_path}")


if __name__ == "__main__":
    main()
