#!/bin/bash
for ((b=1;b<=10;b++))
    do
        python normal_process.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 500 ./model_TL_bg.pth.tar ./data/root_dir --lr 0.1
    done
