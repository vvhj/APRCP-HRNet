# ------------------------------------------------------------------------------
# Copyright (c) CUMT
# Licensed under the MIT License.
# Written by Dongjingdian Liu (3357816826@qq.com)
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
from core.function import validate
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
import heapq

def minremainindex(lis,top=1,descending=True):
    list1 = lis.clone().cpu().numpy().tolist()
    if descending:
        Index = list(map(list1.index,heapq.nlargest(top, list1)))
    else:
        Index = list(map(list1.index,heapq.nsmallest(top, list1)))
    return Index

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
    parser.add_argument('--percent',
                        help='Purning Percent',
                        type=float,
                        default=0.78)
    parser.add_argument('--save', default='models/pose_coco_prune', type=str, metavar='PATH',
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
    perf_indicator = validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    return perf_indicator

def getprunemodel(percent=0.6,index=0):
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')
    final_output_dir += ('/'+str(index))
    tb_log_dir += ('/'+str(index))
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    #oldmodel = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        #cfg, is_train=False
    #)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        #oldmodel.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
        #oldmodel.load_state_dict(torch.load(model_state_file))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    #oldmodel = torch.nn.DataParallel(oldmodel, device_ids=cfg.GPUS).cuda()
    #print('旧模型: ', model)
    modules = list(model.modules())

    purn_bn = []
    for k,m in enumerate(modules):
        name = m._get_name()
        if name == 'Bottleneck':
            purn_bn.append(k+2)
            purn_bn.append(k+4)
        if name == 'BasicBlock':
            purn_bn.append(k+2)

    # for m in modules:
    #     m = list(m.modules)
    #     for mi in m:
    #         print(1)
    # modules = list(model.modules())
    # model_channel_dir = {}
    # for key in model.modules():
    #     while(len(key._modules)!=0):
    #         model_channel_dir['key._get_name()'] = {}

    #     if (len(key._modules)==0):
    #         print(key)
    #     else:
    #         print(key._get_name())
    #         for keyi in key.modules():
    #             print(keyi)
    total = 0
    i = 0
    for m in model.modules():
        if i not in purn_bn:
            i += 1
            continue
        if isinstance(m,nn.BatchNorm2d):
            total += m.weight.data.shape[0]
            i += 1
        else:
            print("error! plase check lyer type")
    bn = torch.zeros(total)
    index = 0
    i = 0
    for m in model.modules():
        if i not in purn_bn:
            i += 1
            continue
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
            i += 1
    y,i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    ncfg = []
    allcfg = []
    allcfg_mask = []
    ncfg_mask = []
    old_index = []
    for k,m in enumerate(model.modules()):
        if isinstance(m,nn.BatchNorm2d):
            if k not in purn_bn:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                allcfg.append(int(mask.shape[0]))
                allcfg_mask.append(mask.clone())
                #ncfg.append(int(mask.shape[0]))
                #ncfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], mask.shape[0]))
                continue
            weight_copy = m.weight.data.abs().clone()
            #weight_temp = weight_copy.gt(float(weight_copy.data.max)).float()
            mask = weight_copy.gt(thre).float().cuda()
            minchannels = int(weight_copy.size(0)*(1-percent)/4)
            if (int(torch.sum(mask))<= minchannels):
                index = minremainindex(weight_copy,minchannels)
                for iii in index:
                    mask[iii] = 1

            pruned = pruned + mask.shape[0] - max(int(torch.sum(mask)),1)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            ncfg.append(max(1,int(torch.sum(mask))))
            ncfg_mask.append(mask.clone())
            allcfg.append(max(1,int(torch.sum(mask))))
            allcfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m,nn.MaxPool2d):
            ncfg.append('M')
    pruned_ratio = pruned/total
    print('Pre-processing Successful!')

    #perf_indicator = test(model)
    print("new cfg:")
    print(ncfg)
    newmodel = eval('models.'+'purnpose_hrnet'+'.get_pose_net')(
        cfg,ncfg, is_train=False
    )#PosePurnHighResolutionNet(cfg,ncfg)
    newmodel = torch.nn.DataParallel(newmodel, device_ids=cfg.GPUS).cuda()
    newmodules = list(newmodel.modules())
    print("newmodelcreate!")
    testncfg = []
    for k,m in enumerate(newmodel.modules()):
        if isinstance(m,nn.BatchNorm2d):
            if k not in purn_bn:
                continue
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            testncfg.append(int(mask.shape[0]))
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m,nn.MaxPool2d):
            testcfg.append('M')
    if testncfg == ncfg:
        print("check step1 succesful")
    else:
        print("error")
    
    print("init newmodel weight:")
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(testncfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = allcfg_mask[layer_id_in_cfg]
    conv_count = 0
    for layer_id in range(len(modules)):
        m0 = modules[layer_id]
        m1 = newmodules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            if layer_id not in purn_bn:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(allcfg_mask):  # do not change in Final FC
                    end_mask = allcfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(allcfg_mask):  # do not change in Final FC
                    end_mask = allcfg_mask[layer_id_in_cfg]
        elif isinstance(m0,nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if  ((layer_id+1) in purn_bn): #单输出层和输出输入层剪枝
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
                #w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                if ((layer_id-1) in purn_bn) or ((layer_id-2) in purn_bn): #输入层剪枝
                    w1 = w1[:,idx0.tolist(),:,:].clone()
                    print('Iut shape {:d}.'.format(idx0.size))
                else:
                    print('Iut shape {:d}.'.format(w1.shape[1]))
                #w2 = w1[idx1.tolist(), :, :, :].clone()
                print('Out shape {:d}.'.format(idx1.size))
                m1.weight.data = w1.clone()
                continue
            if ((layer_id-1) in purn_bn) or ((layer_id-2) in purn_bn): #单输入层剪枝
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:,idx0.tolist(),:,:].clone()
                m1.weight.data = w1.clone()
                print('Iut shape {:d}.'.format(idx0.size))
                print("Out shape {:d}".format(w1.shape[0]))
                continue
            m1.weight.data = m0.weight.data.clone()
    torch.save({'cfg': cfg,'ncfg':ncfg ,'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned'+'_pecnet'+str(percent)+'.pth.tar'))
    #torch.save({'cfg': cfg ,'state_dict': oldmodel.state_dict()}, os.path.join(args.save, 'oldpruned.pth.tar'))
    print('newmodelsaved')
    print("testnewmodel:")
    model = newmodel
    return model
def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    percent = args.percent#0.6

    model = getprunemodel(percent)
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models/purnpose_hrnet.py'),
        final_output_dir)
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
   # writer_dict['writer'].add_graph(model, (dump_input, ))
    logger.info(get_model_summary(model, dump_input))
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


    
if __name__ == '__main__':
    main()

