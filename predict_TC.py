import argparse
import os
import shutil
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet
i=10
os.mkdir(r'E:\研究生文件\cgcnn-master\predict\model_%s'%i)
old_path_1=r'E:\研究生文件\cgcnn-master\model_best.pth.tar'
old_path_2=r'E:\研究生文件\cgcnn-master\model_TL.pth.tar'
model_path=r'E:\研究生文件\cgcnn-master\predict\model_%s\model.pth.tar'%i
results_path=r'E:\研究生文件\cgcnn-master\predict\model_%s'%i
shutil.copy(old_path_2,model_path)

if os.path.isfile(model_path):
    print("=> loading model params '{}'".format(model_path))
    model_checkpoint = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(model_path))


cuda = torch.cuda.is_available()
cifpath = r'E:\研究生文件\cgcnn-master\data\cif_ebg'
dataset = CIFData(cifpath)
collate_fn = collate_pool
test_loader = DataLoader(dataset, batch_size=36, shuffle=True,
                         num_workers=0, collate_fn=collate_fn,
                         pin_memory=cuda)
structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]
model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                            atom_fea_len=model_args.atom_fea_len,
                            n_conv=model_args.n_conv,
                            h_fea_len=model_args.h_fea_len,
                            n_h=model_args.n_h,
                            classification=False)
model.cuda()


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


normalizer = Normalizer(torch.zeros(3))
checkpoint = torch.load(model_path,
                        map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
normalizer.load_state_dict(checkpoint['normalizer'])


def predict(val_loader, model):
    test_preds = []
    test_cif_ids = []
    model.eval()
    for i, (input, target, batch_cif_ids) in tqdm(enumerate(val_loader)):
        with torch.no_grad():
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        output = model(*input_var)
        test_pred = normalizer.denorm(output.data.cpu())
        test_preds += test_pred.view(-1).tolist()
        test_cif_ids += batch_cif_ids
        import csv
        with open(results_path+'/results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred in zip(test_cif_ids,
                                    test_preds):
                writer.writerow((cif_id, pred))


predict(test_loader, model)