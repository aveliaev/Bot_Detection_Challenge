#!/bin/sh
cd "$(dirname "$0")/.."
while pgrep -f "train_sequence.py ablation" > /dev/null; do sleep 30; done
python research7/integrate_nn.py > research7/int_E7nn.log 2>&1
