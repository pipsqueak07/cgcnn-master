#!/bin/bash
for (( a = 1; a <= 10; a++ ))
do
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 500  ./data/root_dir/ --lr 0.01
done