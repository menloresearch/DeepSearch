# Brain Rotting Multiple GPU Workflow for Dummies

## Problem: Working with Multiple GPUs Without Race Conditions

Running multiple training processes on different GPUs can lead to:

- Output directory conflicts
- Checkpoint corruption
- Resource contention
- Difficult debugging and tracking

This guide gives you dead simple solutions using only basic scripts.

## Directory Structure for Sanity

First, set up a clean directory structure to keep runs separate:

```
project/
├── scripts/
│   ├── train_gpu0.sh
│   ├── train_gpu1.sh 
│   └── monitor_gpus.sh
├── src/
│   └── train.py
└── runs/
    ├── gpu0/  # Training on GPU 0
    │   ├── checkpoints/
    │   └── logs/
    └── gpu1/  # Training on GPU 1
        ├── checkpoints/
        └── logs/
```

## Simple Shell Scripts for GPU Management

### 1. Dedicated GPU Training Script (train_gpu0.sh)

```bash
#!/bin/bash

# Assign this process to GPU 0 only
export CUDA_VISIBLE_DEVICES=0

# Create timestamped run directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/gpu0/${TIMESTAMP}"
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs

# Run with output redirect to log file
python src/train.py \
  --output_dir $OUTPUT_DIR \
  --batch_size 32 \
  --learning_rate 1e-4 \
  > $OUTPUT_DIR/logs/training.log 2>&1
```

### 2. Second GPU Script (train_gpu1.sh)

```bash
#!/bin/bash

# Assign this process to GPU 1 only
export CUDA_VISIBLE_DEVICES=1

# Create timestamped run directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/gpu1/${TIMESTAMP}"
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs

# Run with output redirect to log file
python src/train.py \
  --output_dir $OUTPUT_DIR \
  --batch_size 32 \
  --learning_rate 1e-4 \
  > $OUTPUT_DIR/logs/training.log 2>&1
```

### 3. Simple GPU Monitoring Script (monitor_gpus.sh)

```bash
#!/bin/bash

# Simple GPU monitoring loop with timestamps
while true; do
  clear
  echo "======== $(date) ========"
  nvidia-smi
  sleep 5
done
```

## Checkpoint Management Without Race Conditions

In your `train.py`, implement safe checkpoint saving:

```python
import os
import time
import torch
import shutil
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, step, args):
    """Save checkpoint safely without race conditions"""
    # Get process-specific info for uniqueness
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create temporary directory with unique name
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    temp_dir = checkpoint_dir / f"temp_{pid}_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    
    # Save to temporary location first
    checkpoint_path = temp_dir / "checkpoint.pt"
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    # Create final directory name
    final_dir = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}"
    
    # Atomic rename operation (safer than copying files)
    shutil.move(str(temp_dir), str(final_dir))
    
    # Clean up old checkpoints (keep only last 5)
    checkpoints = sorted([d for d in checkpoint_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("checkpoint_")])
    for old_checkpoint in checkpoints[:-5]:
        shutil.rmtree(old_checkpoint)
    
    print(f"Saved checkpoint to {final_dir}")
    return final_dir
```

## Running Multiple Training Jobs with Different Parameters

Create a parameter sweep script that launches multiple runs with different configs:

```bash
#!/bin/bash
# param_sweep.sh

# Define parameter grid
LEARNING_RATES=("1e-3" "5e-4" "1e-4")
BATCH_SIZES=(16 32 64)

# Loop through parameters and assign to GPUs
GPU=0
for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    # Select GPU using modulo to cycle through available GPUs
    SELECTED_GPU=$(($GPU % 2)) # Assuming 2 GPUs (0 and 1)
    GPU=$((GPU + 1))
    
    # Create run directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_NAME="lr${lr}_bs${bs}"
    OUTPUT_DIR="runs/gpu${SELECTED_GPU}/${RUN_NAME}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR/checkpoints
    mkdir -p $OUTPUT_DIR/logs
    
    # Launch training in background
    echo "Starting run on GPU ${SELECTED_GPU}: lr=${lr}, bs=${bs}"
    CUDA_VISIBLE_DEVICES=$SELECTED_GPU python src/train.py \
      --output_dir $OUTPUT_DIR \
      --batch_size $bs \
      --learning_rate $lr \
      > $OUTPUT_DIR/logs/training.log 2>&1 &
    
    # Wait a bit to stagger the starts
    sleep 10
  done
done

echo "All jobs launched. Monitor with './scripts/monitor_gpus.sh'"
```

## Dead Simple Experiment Tracking Without MLflow

Create a simple CSV logger in your training script:

```python
import csv
from pathlib import Path

class SimpleLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics CSV
        self.metrics_file = self.log_dir / "metrics.csv"
        self.header_written = False
        
        # Keep track of best metrics
        self.best_metrics = {}
    
    def log_metrics(self, metrics, step):
        """Log metrics to CSV file"""
        metrics["step"] = step
        
        # Create or append to CSV
        write_header = not self.metrics_file.exists()
        
        with open(self.metrics_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        
        # Update best metrics
        for key, value in metrics.items():
            if key != "step":
                if key not in self.best_metrics or value < self.best_metrics[key]["value"]:
                    self.best_metrics[key] = {"value": value, "step": step}
        
        # Write best metrics summary
        with open(self.log_dir / "best_metrics.txt", 'w') as f:
            for key, data in self.best_metrics.items():
                f.write(f"Best {key}: {data['value']} (step {data['step']})\n")
```

## Finding and Comparing Results

Create a simple results aggregation script:

```bash
#!/bin/bash
# aggregate_results.sh

echo "Run Directory,Best Loss,Best Accuracy,Training Time"

find runs/ -name "best_metrics.txt" | while read metrics_file; do
    run_dir=$(dirname "$metrics_file")
    best_loss=$(grep "Best loss" "$metrics_file" | cut -d' ' -f3)
    best_acc=$(grep "Best accuracy" "$metrics_file" | cut -d' ' -f3)
    
    # Get training time from log
    log_file="$run_dir/logs/training.log"
    start_time=$(head -n 1 "$log_file" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
    end_time=$(tail -n 10 "$log_file" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -n 1)
    
    echo "$run_dir,$best_loss,$best_acc,$start_time-$end_time"
done | sort -t',' -k2n
```

## Simple Visualization Without External Tools

Create a basic plotting script using matplotlib:

```python
# plot_results.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find all metrics.csv files
metrics_files = glob.glob("runs/**/metrics.csv", recursive=True)

plt.figure(figsize=(12, 8))

# Plot each run
for metrics_file in metrics_files:
    run_name = Path(metrics_file).parent.name
    df = pd.read_csv(metrics_file)
    
    plt.plot(df['step'], df['loss'], label=f"{run_name} - loss")
    
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_comparison.png')
plt.close()

# Create accuracy plot if available
plt.figure(figsize=(12, 8))
for metrics_file in metrics_files:
    run_name = Path(metrics_file).parent.name
    df = pd.read_csv(metrics_file)
    
    if 'accuracy' in df.columns:
        plt.plot(df['step'], df['accuracy'], label=f"{run_name} - accuracy")
    
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
```

## Process Management and GPU Allocation

Create a script to check GPU usage and allocate new jobs:

```bash
#!/bin/bash
# allocate_gpu.sh

# This script checks GPU usage and returns the index of the least utilized GPU
LEAST_BUSY_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | 
                 sort -t',' -k2n | 
                 head -n 1 | 
                 cut -d',' -f1)

echo $LEAST_BUSY_GPU
```

## Tips for Avoiding Race Conditions

1. **Always use unique output directories for each run**:
   - Include timestamp, GPU ID, and PID in directory names
   - Never share output directories between processes

2. **For checkpoint saving**:
   - Save to temp directory first
   - Use atomic operations like `shutil.move()` for final placement
   - Don't depend on file locks (often unreliable with network filesystems)

3. **For data loading**:
   - Use different random seeds per process
   - Set `num_workers` appropriately (2-4 per GPU usually works well)
   - Add process-specific buffer to avoid filesystem contention

4. **For logging**:
   - Each process should write to its own log file
   - Use timestamps in log entries
   - Include GPU ID and PID in log messages

## Quick Commands Reference

```bash
# Start training on GPU 0
./scripts/train_gpu0.sh

# Start training on GPU 1
./scripts/train_gpu1.sh

# Run parameter sweep across GPUs
./scripts/param_sweep.sh

# Monitor GPU usage
./scripts/monitor_gpus.sh

# Find GPU with lowest utilization
BEST_GPU=$(./scripts/allocate_gpu.sh)
echo "Least busy GPU is: $BEST_GPU"

# Aggregate results into CSV
./scripts/aggregate_results.sh > results_summary.csv

# Generate comparison plots
python scripts/plot_results.py
```

Remember: The simplest solution is usually the most maintainable. Keep your scripts straightforward, make each run independent, and use filesystem organization to avoid conflicts.

# TODO: Replace print statements with loguru logging for better debugging and log file management
