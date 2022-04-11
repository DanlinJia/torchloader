import argparse
import os
import random
import shutil
import time
import datetime
from time import strftime
import logging
import warnings
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# from models.vanilla_resnet_slow import *
# resnet34S = resnet34
# resnet18S = resnet18
# from models.vanilla_resnet import *
# from models.small_model import Small_Net, Small_Net_2
# from models.resnet20_imagenet import resnet20
# from models.vgg_grasp import vgg11, vgg13, vgg16, vgg19

from models.vanilla_resnet_slow import resnet18S, resnet34S
from models.vanilla_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg_grasp import *
from models.small_model import *
from models.resnet20_imagenet import resnet20
from models.builder import build_model

import yaml
import numpy as np

import torch.autograd.profiler as profiler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    # choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--iteration', default=100, type=int, )
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='define optimizer')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')

parser.add_argument('--trace', type=str, default='test.csv',
                    help='plot trace')
parser.add_argument('--profile', action='store_true',
                    help='Use profiler')
parser.add_argument('--profile-epochs', default=0, type=int,
                    help='Use profiler')
parser.add_argument('--profile-name', type=str, default='test',
                    help='name of profile test')
parser.add_argument('--record-shapes', action='store_true',
                    help='define lr scheduler')
parser.add_argument('--profile-memory', action='store_true',
                    help='Use profiler')
parser.add_argument('--use-cuda', action='store_true',
                    help='define lr scheduler')
parser.add_argument('--profile-batches', default=-1, type=int,
                    help='Use profiler')
parser.add_argument('--app-id', default=1, type=int, help="Application Unique ID")



best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(f"ngpus_per_node={ngpus_per_node}")
    if args.multiprocessing_distributed:
        print("multiprocessing_distributed")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("local mod")
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        local_rank = args.gpu
        if 'SLURM_NPROCS' in os.environ:
            ngpus_per_node = torch.cuda.device_count()
            args.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node # compute world_size for multi-node
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR'] # get master addr
            args.rank = int(os.environ['SLURM_PROCID']) * ngpus_per_node + local_rank # global rank
            # args.node_list = os.environ["SLURM_NODELIST"] # All node you are using
            print("slurm cluster constructed: \n Divice per node: {}, world size: {}, url: {}, local_rank/gpu_id: {}, globale_rank: {}".format(\
                    ngpus_per_node, args.world_size, args.dist_url,local_rank, args.rank ))
        else:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=30))
    

    print("=> creating model '{}'".format(args.arch))
    model = build_model(args.arch, args.depth)

    # if args.arch == "resnet":
    #     if args.depth == 18:
    #         model = resnet18()
    #     elif args.depth == 19:
    #         model = resnet18S()
    #     elif args.depth == 50:
    #         model = resnet50()
    #     elif args.depth == 20:
    #         model = resnet20()
    #     elif args.depth == 34:
    #         model = resnet34()
    #     elif args.depth == 35:
    #         model = resnet34S()
    #     elif args.depth == 101:
    #         model = resnet101()
    #     elif args.depth == 152:
    #         model = resnet152()
    #     else:
    #         sys.exit("resnet doesn't implement those depth!")
    # elif args.arch == "vgg":
    #     if args.depth == 11:
    #         model = vgg11()
    #     elif args.depth == 13:
    #         model = vgg13()
    #     elif args.depth == 16:
    #         model = vgg16()
    #     elif args.depth == 19:
    #         model = vgg19()
    #     elif args.depth == 5:
    #         model = vgg5()
    #     elif args.depth == 6:
    #         model = vgg6()
    #     elif args.depth == 7:
    #         model = vgg7()
    #     elif args.depth == 51:
    #         model = vgg51()
    #     elif args.depth == 61:
    #         model = vgg61()
    #     elif args.depth == 71:
    #         model = vgg71()
    #     elif args.depth == 52:
    #         model = vgg52()
    #     elif args.depth == 62:
    #         model = vgg62()
    #     elif args.depth == 72:
    #         model = vgg72()
    #     else:
    #         sys.exit("vgg doesn't implement those depth!")

    # elif args.arch == "small":
    #     if args.depth == 2:
    #         model = small2()
    #     elif args.depth == 3:
    #         model = small3()
    #     elif args.depth == 4:
    #         model = small4()
    #     elif args.depth == 5:
    #         model = small5()
    #     elif args.depth == 6:
    #         model = small6()
    #     elif args.depth == 7:
    #         model = small7()
    #     elif args.depth == 8:
    #         model = small8()
    #     elif args.depth == 9:
    #         model = small9()
    #     elif args.depth == 10:
    #         model = small10()
    #     else:
    #         sys.exit("small models doesn't implement those depth!")
    # else:
    #     sys.exit("unknown network")
    #print(model)


    expriment_name = "{}{}_{}_{}_{}_app{}".format(args.arch, args.depth, args.batch_size, args.workers, ngpus_per_node ,args.app_id)
    args.trace = os.path.join(args.trace, expriment_name)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print("1")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model = torch.nn.DataParallel(model)
            model.cuda()
            print("2")
        else:
            model = torch.nn.DataParallel(model).cuda()
            print("3")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

#################################################  warmup  ###############
    optimizer_init_lr = args.lr

    optimizer = None
    if (args.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif (args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

        #  scheduler
########################################################################
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [60, 120]
        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_milestones, gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")


###########################################################################

    # training
    if args.profile == False:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            print(f"gpu:{args.rank} starts epoch: {epoch}")
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # adjust learning rate
            scheduler.step()

            epoch_end = time.time()
            print("epoch_time: {}".format(epoch_end - epoch_start))

    else:
        if args.profile_batches == -1:
            args.profile_batches = len(train_loader)

        with profiler.profile(record_shapes=args.record_shapes, profile_memory=args.profile_memory, use_cuda=args.use_cuda) as prof:
            with profiler.record_function(expriment_name):
                for epoch in range(1, args.profile_epochs + 1):
                    epoch_start = time.time()
                    if args.distributed:
                        train_sampler.set_epoch(epoch)

                    # train for one epoch
                    train(train_loader, model, criterion, optimizer, epoch, args)

                    # adjust learning rate
                    scheduler.step()

                    epoch_end = time.time()
                    print("epoch_time: {}".format(epoch_end - epoch_start))

        # save profile results
        profile_name = args.profile_name
        trace_name = "{}/{}_{}.json".format(args.trace, expriment_name, args.rank)
        prof.export_chrome_trace(trace_name)


    # for epoch in range(1, args.epochs + 1):
    #     epoch_start = time.time()
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #
    #     # train for one epoch
    #     idx_loss_dict = train(train_loader, model, criterion, optimizer, epoch, args, scheduler, teacher=None)
    #
    #     # adjust learning rate
    #     scheduler.step()
    #
    #     epoch_end = time.time()
    #     print("epoch_time: {}".format(epoch_end-epoch_start))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    update_time = AverageMeter()
    losses = AverageMeter()
    datatrans_time = AverageMeter()
    lossupdate_time = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i != -1 and i == args.iteration:
            break
        # measure data loading time
        dataloading_ts = time.time()
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # measure data transfer time
        datatransfer_ts = time.time()
        output = model(input)
        ce_loss = criterion(output, target)
        forward_ts = time.time()
        # get loss from GPU, deprecated for performance concern
        #losses.update(ce_loss.item(), input.size(0))
        loss_ts = time.time()
        # compute gradient and do SGD step
        # replace optimizer.zero_grad(), taking less memory oprations
        for param in model.parameters():
            param.grad = None
        ce_loss.backward()
        backward_ts = time.time()
        # update
        optimizer.step()
        update_ts = time.time()
        # measure elapsed time with breakdown
        start = end
        end = time.time()
        batch_time.update(end - start)
        data_time.update(dataloading_ts - start)
        datatrans_time.update(datatransfer_ts - dataloading_ts)
        forward_time.update(forward_ts - datatransfer_ts)
        lossupdate_time.update(loss_ts - forward_ts)
        backward_time.update(backward_ts - loss_ts)
        update_time.update(update_ts- backward_ts)
        if i % args.print_freq == 0:
            print("iteration {} takes {}".format(i, end-start))
        # if i % args.print_freq == 0:
        #     for param_group in optimizer.param_groups:
        #         current_lr = param_group['lr']
        #     print('Epoch: [{0}][{1}/{2}] [lr={3}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DataLoading {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
        #           'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
        #           'Update {update_time.val:.3f} ({update_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         .format(epoch, i, len(train_loader), current_lr, batch_time=batch_time,
        #         loss=losses, data_time=data_time, forward_time=forward_time, backward_time=backward_time, update_time=update_time))
        #     print("cross_entropy loss: {}".format(ce_loss))
    df = pd.DataFrame(columns=["iteration","dataloading","datatransfer", "forward","lossupdate", "backward","update","loss"])
    df["iteration"] = np.array(batch_time.tracker)
    df["dataloading"] = np.array(data_time.tracker)
    df["datatransfer"] = np.array(datatrans_time.tracker)
    df["forward"] = np.array(forward_time.tracker)
    df["lossupdate"] = np.array(lossupdate_time.tracker)
    df["backward"] = np.array(backward_time.tracker)
    df["update"] = np.array(update_time.tracker)
    #df["loss"] = np.array(losses.tracker[:len(df)])
    if not os.path.exists(args.trace):
        os.system("mkdir -p {}".format(args.trace))
    df.to_csv(args.trace + "/device_{}.csv".format(args.rank), index=False)

def train_profile(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    update_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i==args.iteration:
            break

        # measure data loading time
        dataloading_ts = time.time()
        data_time.update(dataloading_ts - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)

        ce_loss = criterion(output, target)

        forward_ts = time.time()
        forward_time.update(forward_ts - dataloading_ts)
        losses.update(ce_loss.item(), input.size(0))

        loss_ts = time.time()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        ce_loss.backward()
        backward_ts = time.time()
        backward_time.update(backward_ts - loss_ts)

        optimizer.step()
        update_time.update(time.time() - backward_ts)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     for param_group in optimizer.param_groups:
        #         current_lr = param_group['lr']
        #     print('Epoch: [{0}][{1}/{2}] [lr={3}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DataLoading {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
        #           'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
        #           'Update {update_time.val:.3f} ({update_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         .format(epoch, i, len(train_loader), current_lr, batch_time=batch_time,
        #         loss=losses, data_time=data_time, forward_time=forward_time, backward_time=backward_time, update_time=update_time))
        #     print("cross_entropy loss: {}".format(ce_loss))
    df = pd.DataFrame(columns=["iteration","dataloading","forward","backward","update","loss"])
    df["iteration"] = np.array(batch_time.tracker)
    df["dataloading"] = np.array(data_time.tracker)
    df["forward"] = np.array(forward_time.tracker)
    df["backward"] = np.array(backward_time.tracker)
    df["update"] = np.array(update_time.tracker)
    df["loss"] = np.array(losses.tracker)
    if not os.path.exists(args.trace):
        os.system("mkdir -p {}".format(args.trace))
    df.to_csv(args.trace + "/device_{}.csv".format(args.rank), index=False)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.tracker = []

    def update(self, val, n=1):
        self.val = val
        self.tracker.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

