#!/bin/bash
for ((b=1;b<=10;b++))
    do
        python normal_process.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150 ./model_TL.pth.tar ./data/phonon_bg
    done
python normal_process.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150 ./model_best.pth.tar ./data/phonon_bg --dt init --lr 0.01
