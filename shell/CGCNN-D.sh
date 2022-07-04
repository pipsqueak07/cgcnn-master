#!/bin/bash
for ((a=1;a<=5;a++))
  do
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150  ./data/phonon_bg
  done

