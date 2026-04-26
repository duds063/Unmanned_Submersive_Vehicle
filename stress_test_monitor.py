#!/usr/bin/env python3
"""
Stress Test Monitoring Script
Monitors replay file generation and performance metrics during benchmark
"""

import os
import json
import time
import glob
from pathlib import Path
from datetime import datetime

def format_size(bytes_val):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"

def count_frames(jsonl_path):
    """Count frames in a JSONL file"""
    try:
        with open(jsonl_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0

def monitor_stress_test():
    replay_dir = Path("training_runs/replays")
    
    if not replay_dir.exists():
        print(f"Replay directory not found: {replay_dir}")
        return
    
    print("\n" + "="*70)
    print(f"  STRESS TEST MONITOR — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Count files
    meta_files = list(replay_dir.glob("*.meta.json"))
    jsonl_files = list(replay_dir.glob("*.jsonl"))
    
    print(f"\n📊 File Counts:")
    print(f"   JSONL frame files: {len(jsonl_files)}")
    print(f"   Metadata files: {len(meta_files)}")
    print(f"   Total pairs: {len(meta_files)}")
    
    # Calculate total size and frames
    total_size = 0
    total_frames = 0
    largest_file = None
    largest_size = 0
    
    print(f"\n📈 Top 10 Largest Replay Files:")
    file_stats = []
    
    for jsonl_path in jsonl_files:
        try:
            size = jsonl_path.stat().st_size
            frames = count_frames(str(jsonl_path))
            total_size += size
            total_frames += frames
            
            file_stats.append({
                'name': jsonl_path.stem,
                'size': size,
                'frames': frames,
                'avg_frame_size': size / frames if frames > 0 else 0
            })
            
            if size > largest_size:
                largest_size = size
                largest_file = jsonl_path.stem
        except Exception as e:
            print(f"   Error reading {jsonl_path.name}: {e}")
    
    # Sort by size
    file_stats.sort(key=lambda x: x['size'], reverse=True)
    
    for i, stat in enumerate(file_stats[:10], 1):
        print(f"   {i:2d}. {stat['name'][:45]:45s} {format_size(stat['size']):>8s} ({stat['frames']:5d} frames)")
    
    # Summary stats
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Total data: {format_size(total_size)}")
    print(f"   Total frames: {total_frames:,}")
    if total_frames > 0:
        print(f"   Avg frame size: {total_size/total_frames:.1f} bytes")
    if len(jsonl_files) > 0:
        print(f"   Avg file size: {total_size/len(jsonl_files)}")
    
    # Expected vs actual
    expected_runs = 30  # 10 trials × 3 controllers
    expected_files = expected_runs * 2  # jsonl + meta
    progress = (len(meta_files) / expected_runs) * 100
    
    print(f"\n⏳ Progress:")
    print(f"   Completed: {len(meta_files)}/{expected_runs} runs ({progress:.1f}%)")
    print(f"   Files created: {len(meta_files) + len(jsonl_files)}/{expected_files}")
    
    # Playback performance estimation
    if total_frames > 0:
        print(f"\n🎬 Playback Performance Estimate:")
        avg_fps = 60
        total_playback_hours = (total_frames * 0.01) / 3600  # 0.01s per frame
        print(f"   Average payload size: {total_size/total_frames:.0f} bytes/frame")
        print(f"   @ 60Hz, total playback time: {total_playback_hours:.1f} hours")
        print(f"   Bandwidth needed @ 60Hz: {(total_size/total_frames)*60/1024/1024:.1f} MB/s")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    monitor_stress_test()
