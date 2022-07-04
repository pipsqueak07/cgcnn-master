#!/bin/bash
for ((a=1;a<=10;a++))
do
    python main.py --train-ratio 0.7 --val-ratio 0 --test-ratio 0.3 --optim Adam --epochs 70  ./data/ebg
    for ((b=1;b<=10;b++))
    do
        python normal_process.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 150 ./model_TL_bg.pth.tar ./data/phonon_bg --lr 0.1
    done
done

$ python main.py --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2 --optim Adam --epochs 500  ./data/debye_tl/ --lr 0.01
