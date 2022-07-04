#!/bin/bash
python normal_process_Kfold.py --optim Adam --epochs 200 ./model_xenonpy.pth.tar  ./data/Cv_TL/ --lr 0.05
                                                                                  ./data/specific_heat_TL --lr 0.05
