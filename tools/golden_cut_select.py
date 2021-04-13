# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import torch.nn as nn
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import numpy as np
import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate,validate_select
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from models.pose_resnet import Bottleneck,BasicBlock
from models.purnpose_hrnet import PosePurnHighResolutionNet
from dataset import coco
import dataset
import models
import json
from purning_criterion import getpruneffects,getpruneffects_v2,getpruneffects_v3

# v3 0.5514472610784643
# v2 0.47681901943061233
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--save', default='MPIIHRNET48_sns', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

    args = parser.parse_args()

    return args

def test(model,cfg,final_output_dir,tb_log_dir):
    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    #model.eval()
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    perf_indicator = validate_select(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    return perf_indicator

def fic(percent,ap,acc,maxap,maxacc):
    return acc/maxacc + ap/maxap + percent

def main():
    epsilon = 1e-3
    alpha = 0.618
    a = 0
    b = 1
    args = parse_args()
    update_config(cfg, args)
    datadir = {}
    datadir["percent"] = []
    datadir["Perf_indicator"] = []
    datadir["Acc"] = []
    max_perf,max_acc = getpruneffects(0,"original")
    per = 0
    count = 0
    max_count = 45
    while b - a > 0:
        lam = a + (1 - alpha) * (b - a)
        mu = a + alpha * (b - a)
        print("count:{},a:{},b:{},lam:{},mu:{}".format(count,a,b,lam,mu))
        Per_lam,Acc_lam = getpruneffects(lam,int(lam*100000))
        datadir["percent"].append(lam)
        datadir["Perf_indicator"].append(Per_lam)
        datadir["Acc"].append(Acc_lam)
        Per_mu,Acc_mu = getpruneffects(mu,int(mu*100000))
        datadir["percent"].append(mu)
        datadir["Perf_indicator"].append(Per_mu)
        datadir["Acc"].append(Acc_mu)
        F_lam = fic(lam,Per_lam,Acc_lam,max_perf,max_acc)
        F_mu = fic(mu,Per_mu,Acc_mu,max_perf,max_acc)
        if (b - a < epsilon) or (count > max_count):
            percent = (b+a)/2
            Perf_indicator , Acc = getpruneffects(percent,"final_percent{}".format(int(percent*100000)))
            datadir["percent"].append(percent)
            datadir["Perf_indicator"].append(Perf_indicator)
            datadir["Acc"].append(Acc)
            print(percent)
            break
        elif F_lam > F_mu:
            a = lam
            lam = mu
            mu =  a + alpha * (b - a)
        elif F_lam <= F_mu:
            b = mu
            mu = lam
            lam = a + (1 - alpha) * (b - a)
        count += 1
    with open(os.path.join(args.save,'datav2.json'), 'w') as f:
        json.dump(datadir, f)


if __name__ == '__main__':
    main()
