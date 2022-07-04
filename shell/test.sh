#!/bin/bash
for (( a = 1; a <= 10; a++ ))
do
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --epochs 150 --optim Adam ./data/phonon_bg
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150  ./data/ebg
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150  ./data/debye
    python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150  ./data/Speed\ of\ sound


done