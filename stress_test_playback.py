#!/usr/bin/env python3
"""
Replay Playback Performance Test
Tests playback speed, memory usage, and seek performance with large replays
"""

import json
import time
import psutil
import os
from pathlib import Path
from visualization_player import VisualizationPlayer

def get_process_memory():
    """Get current process memory in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def test_catalog_loading():
    """Test catalog discovery performance"""
    print("\n" + "="*70)
    print("  CATALOG LOADING TEST")
    print("="*70)
    
    start_mem = get_process_memory()
    start_time = time.time()
    
    player = VisualizationPlayer('./training_runs/replays')
    trials = player.list_trials()
    
    elapsed = time.time() - start_time
    end_mem = get_process_memory()
    
    print(f"\nTrials discovered: {len(trials)}")
    print(f"Load time: {elapsed*1000:.1f}ms")
    print(f"Memory used: {end_mem - start_mem:.1f}MB")
    
    # Show breakdown by controller
    by_ctrl = {}
    for t in trials:
        ctrl = t['controller']
        by_ctrl[ctrl] = by_ctrl.get(ctrl, 0) + 1
    
    print(f"Controller breakdown:")
    for ctrl in sorted(by_ctrl.keys()):
        print(f"  {ctrl}: {by_ctrl[ctrl]} trials")
    
    return player, trials

def test_frame_loading(player, trials, num_samples=5):
    """Test loading frames from largest replays"""
    print("\n" + "="*70)
    print("  FRAME LOADING TEST")
    print("="*70)
    
    if not trials:
        print("No trials to test")
        return
    
    # Get largest replays
    sorted_trials = sorted(trials, key=lambda x: x.get('frame_count', 0), reverse=True)
    test_trials = sorted_trials[:num_samples]
    
    for trial in test_trials:
        run_id = trial['run_id']
        frame_count = trial.get('frame_count', 0)
        duration = trial.get('duration_s', 0)
        
        start_mem = get_process_memory()
        start_time = time.time()
        
        player.load_primary(run_id)
        
        elapsed = time.time() - start_time
        end_mem = get_process_memory()
        mem_delta = end_mem - start_mem
        
        print(f"\n  {run_id[:50]}")
        print(f"    Frames: {frame_count} | Duration: {duration:.1f}s | Load: {elapsed*1000:.1f}ms | Δ Memory: {mem_delta:.1f}MB")

def test_playback_speed(player, trials):
    """Test playback at different speeds"""
    print("\n" + "="*70)
    print("  PLAYBACK SPEED TEST")
    print("="*70)
    
    if not trials:
        print("No trials to test")
        return
    
    # Use largest replay
    trial = max(trials, key=lambda x: x.get('frame_count', 0))
    run_id = trial['run_id']
    duration = trial.get('duration_s', 0)
    
    player.load_primary(run_id)
    
    print(f"\nTesting with: {run_id[:50]}")
    print(f"Duration: {duration:.1f}s | Frames: {trial.get('frame_count', 0)}")
    
    for speed in [1.0, 2.0, 5.0]:
        player.speed = speed
        player.playhead_s = 0.0
        player.playing = True
        
        start_time = time.time()
        tick_count = 0
        target_wall_time = duration / speed
        
        # Simulate playback for full duration
        while player.playhead_s < duration and tick_count < 10000:
            dt = 1/60  # 60Hz
            player.tick(dt)
            tick_count += 1
        
        elapsed = time.time() - start_time
        actual_speed = (duration / speed) / elapsed
        
        print(f"  {speed:.1f}x: {elapsed:.2f}s (expected {duration/speed:.2f}s) | Speed ratio: {actual_speed:.2f}x | Ticks: {tick_count}")

def test_seeking_performance(player, trials):
    """Test seek latency at different positions"""
    print("\n" + "="*70)
    print("  SEEK PERFORMANCE TEST")
    print("="*70)
    
    if not trials:
        print("No trials to test")
        return
    
    # Use largest replay
    trial = max(trials, key=lambda x: x.get('frame_count', 0))
    run_id = trial['run_id']
    duration = trial.get('duration_s', 0)
    frame_count = trial.get('frame_count', 0)
    
    player.load_primary(run_id)
    
    print(f"\nTesting with: {run_id[:50]}")
    print(f"Duration: {duration:.1f}s | Frames: {frame_count}")
    
    # Test seeking to various positions
    seek_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\nSeek latency (50 seeks per position):")
    for ratio in seek_positions:
        times = []
        for _ in range(50):
            start = time.time()
            player.seek_ratio(ratio)
            _ = player.current_state()  # Force state build
            times.append((time.time() - start) * 1000)
        
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        min_ms = min(times)
        
        pos_time = ratio * duration
        print(f"  {ratio:.0%} ({pos_time:.2f}s): avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms")

def test_multi_trial_comparison(player, trials):
    """Test envelope calculation with multiple trials"""
    print("\n" + "="*70)
    print("  MULTI-TRIAL COMPARISON TEST")
    print("="*70)
    
    if len(trials) < 3:
        print("Not enough trials for comparison test")
        return
    
    # Select 3 trials
    test_runs = [t['run_id'] for t in trials[:3]]
    
    player.load_primary(test_runs[0])
    player.select_trials(test_runs)
    
    print(f"\nComparing {len(test_runs)} trials")
    for run_id in test_runs:
        print(f"  {run_id[:50]}")
    
    # Test envelope at various positions
    print(f"\nEnvelope calculation latency (20 samples per position):")
    
    for ratio in [0.0, 0.5, 1.0]:
        times = []
        for _ in range(20):
            player.seek_ratio(ratio)
            start = time.time()
            state = player.current_state()
            times.append((time.time() - start) * 1000)
        
        envelope = state.get('comparison_envelope')
        has_envelope = envelope is not None
        
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        
        print(f"  {ratio:.0%}: avg={avg_ms:.3f}ms, max={max_ms:.3f}ms, envelope={has_envelope}")

def main():
    print("\n🔬 USV REPLAY STRESS TEST - PERFORMANCE VALIDATION\n")
    
    # Check if replays exist
    replay_dir = Path("training_runs/replays")
    if not replay_dir.exists() or not list(replay_dir.glob("*.jsonl")):
        print("❌ No replay files found. Run benchmark first:")
        print("   python benchmark_engine.py --trials 10 --controllers lqr mpc rl --max-steps 2000")
        return
    
    try:
        player, trials = test_catalog_loading()
        
        if not trials:
            print("No trials found in catalog")
            return
        
        test_frame_loading(player, trials)
        test_playback_speed(player, trials)
        test_seeking_performance(player, trials)
        test_multi_trial_comparison(player, trials)
        
        print("\n" + "="*70)
        print("  ✅ STRESS TEST COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during stress test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
