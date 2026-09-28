#!/bin/sh
cd "$(dirname "$0")/.."
python research7/train_sequence.py main > research7/train_sequence_main.log 2>&1
python research7/train_sequence.py ablation > research7/train_sequence_ablation.log 2>&1
