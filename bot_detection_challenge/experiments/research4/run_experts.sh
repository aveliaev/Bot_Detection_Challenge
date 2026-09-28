#!/bin/sh
cd "$(dirname "$0")/.."
python research4/experts.py > research4/experts.log 2>&1
python research4/experts_inner.py > research4/experts_inner.log 2>&1
